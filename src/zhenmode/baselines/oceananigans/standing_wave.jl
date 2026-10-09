# Native HydrostaticFreeSurfaceModel wave runner. No diagnostic oracle drives evolution.
started = time_ns()
println("native_start_ns=", started); flush(stdout)
using Pkg, JSON3, NCDatasets, CUDA, Oceananigans
using Oceananigans.Grids: MutableVerticalDiscretization, on_architecture
using Oceananigans.Operators: Δrᶜᶜᶜ
using Oceananigans.TimeSteppers: time_step!

config = JSON3.read(read(ARGS[1],String))
c = config.contract
o = c.oceananigans_options
@assert string(pkgversion(Oceananigans)) == o.version
@assert CUDA.functional()
CUDA.allowscalar(false)
Nx, Ny, Nz = Int(c.nx), Int(c.ny), Int(c.nz)
Δx, Δy = c.Lx_m/Nx, c.Ly_m/Ny
# Native z-star faces are bottom-to-top. Output carries actual native thicknesses.
z_faces = haskey(c, :benchmark) ? reverse(vcat(0.0, -cumsum(Float64.(c.initial_h_m)))) :
          [-c.H_m, -5c.H_m/6, -c.H_m/2, -c.H_m/6, 0.0]
grid = RectilinearGrid(GPU(), Float64; size=(Nx,Ny,Nz), topology=(Periodic,Bounded,Bounded),
                       x=(0,c.Lx_m), y=(0,c.Ly_m), z=MutableVerticalDiscretization(z_faces))
equation_of_state = LinearEquationOfState(Float64; thermal_expansion=o.thermal_expansion,
                                         haline_contraction=o.haline_contraction)
buoyancy = SeawaterBuoyancy(Float64; equation_of_state, gravitational_acceleration=c.gravity)
free_surface = SplitExplicitFreeSurface(grid; gravitational_acceleration=c.gravity, substeps=Int(o.requested_substeps))
rotation = c.f == 0 ? nothing : FPlane(Float64; f=c.f)
model = HydrostaticFreeSurfaceModel(grid; free_surface, buoyancy, coriolis=rotation, closure=nothing,
                                   momentum_advection=VectorInvariant(), tracer_advection=Centered(order=2),
                                   timestepper=Symbol(o.timestepper), vertical_coordinate=ZStarCoordinate(),
                                   tracers=(:T,:S,:physical_T,:physical_S))
# Native linear EOS uses anomalies; two advected physical witnesses keep the
# nonzero-constant tracer control informative and expose representation differences.
η₀(x,y,z) = c.amplitude_m * cos(2π*x/c.Lx_m) * sin(π/Nx)/(π/Nx)
if haskey(c, :benchmark)
    initial = NCDataset("initial-oceananigans.nc", "r")
    try
        Ti, Si, Ui = Array(initial["T"][:,:,:]), Array(initial["S"][:,:,:]), Array(initial["u"][:,:,:])
        set!(model; η=Array(initial["eta"][:,:]), u=Ui, v=0,
             T=Ti.-c.T_C, S=Si.-c.S_psu, physical_T=Ti, physical_S=Si)
    finally
        close(initial)
    end
else
    set!(model; η=η₀, T=0, S=0, physical_T=c.T_C, physical_S=c.S_psu)
end
@assert isnothing(rotation) || model.coriolis.f == c.f
CUDA.synchronize()
@assert model.timestepper.Nstages == 3
end_time = haskey(c,:duration_s) ? c.duration_s : c.period_s
steps = haskey(config,:pilot_steps) ? Int(config.pilot_steps) : Int(end_time/c.dt)
every = Int(c.output_s/c.dt)
Ns = 1 + steps ÷ every
host(field) = Array(interior(field))

output = NCDataset("native.nc","c")
for (name,n) in (("x",Nx),("y",Ny),("yf",Ny+1),("z",Nz),("time",Ns))
    defDim(output,name,n)
end
for (name,field) in (("x_eta",model.free_surface.displacement),("x_u",model.velocities.u),("x_v",model.velocities.v))
    defVar(output,name,Float64,("x",))[:] = vec(Array(xnodes(field)))
end
for (name,field,axis) in (("y_eta",model.free_surface.displacement,"y"),("y_u",model.velocities.u,"y"),("y_v",model.velocities.v,"yf"))
    defVar(output,name,Float64,(axis,))[:] = vec(Array(ynodes(field)))
end
defVar(output,"time",Float64,("time",))
defVar(output,"eta",Float64,("x","y","time"))
defVar(output,"native_zstar_scale",Float64,("x","y","time"))
for name in ("h","T","S","T_anomaly","S_anomaly","u")
    defVar(output,name,Float64,("x","y","z","time"))
end
defVar(output,"v",Float64,("x","yf","z","time"))
output.attrib["vertical_order"] = "bottom_to_top"
output.attrib["model_description"] = sprint(show,model)
output.attrib["coriolis_f_s_1"] = c.f
output.attrib["substepping"] = sprint(show,model.free_surface.substepping)
output.attrib["reference_temperature_C"] = c.T_C
output.attrib["reference_salinity_psu"] = c.S_psu

function snapshot!(n)
    CUDA.synchronize()
    g = on_architecture(CPU(),model.grid)
    # Copying a grid constructs fresh mutable-coordinate storage. Read scaling
    # from the evolving device grid itself, never from the reconstructed copy.
    coordinate = on_architecture(CPU(), model.grid.z)
    scale = Array(coordinate.σᶜᶜⁿ[1:Nx, 1:Ny, 1])
    h = [Δrᶜᶜᶜ(i,j,k,g) * scale[i,j] for i in 1:Nx,j in 1:Ny,k in 1:Nz]
    output["time"][n] = model.clock.time
    output["eta"][:,:,n] = host(model.free_surface.displacement)[:,:,1]
    output["native_zstar_scale"][:,:,n] = scale
    output["h"][:,:,:,n] = h
    for (name,field) in (("T",model.tracers.physical_T),("S",model.tracers.physical_S),
                         ("T_anomaly",model.tracers.T),("S_anomaly",model.tracers.S),
                         ("u",model.velocities.u),("v",model.velocities.v))
        values=host(field)
        @assert all(isfinite,values) "Nonfinite native state"
        output[name][:,:,:,n] = values
    end
    NCDatasets.sync(output)
end
snapshot!(1)
initialization_s = (time_ns()-started)/1e9
println("initialization_s=",initialization_s); flush(stdout)
tick=time_ns()
try
    for step in 1:steps
        time_step!(model,c.dt)
        CUDA.synchronize()
        if step % every == 0
            snapshot!(1+step÷every)
            println("accepted_step=",step," simulation_seconds=",model.clock.time,
                    " integration_s=",(time_ns()-tick)/1e9); flush(stdout)
        end
    end
finally
    output.attrib["integration_s"]=(time_ns()-tick)/1e9
    output.attrib["initialization_s"]=initialization_s
    output.attrib["accepted_steps"]=model.clock.iteration
    close(output)
end
