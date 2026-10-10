using Pkg, JSON3, CUDA, Oceananigans
@assert CUDA.functional() "Oceananigans requires functional CUDA"
packages = Dict(string(id) => Dict("name"=>p.name,"version"=>string(p.version),
                                  "tree_hash"=>string(p.tree_hash),"source"=>p.source)
                for (id,p) in Pkg.dependencies())
oa = only(p for p in values(packages) if p["name"]=="Oceananigans")
actual_tree = bytes2hex(Pkg.GitTools.tree_hash(oa["source"]))
info = Dict("actual_upstream_tree_hash"=>actual_tree,"julia_version"=>string(VERSION),"julia_executable"=>joinpath(Sys.BINDIR,"julia"),
            "oceananigans_version"=>string(pkgversion(Oceananigans)),
            "cuda_version"=>string(pkgversion(CUDA)),"device"=>string(CUDA.device()),
            "packages"=>packages,"runtime"=>sprint(CUDA.versioninfo))
isfile(ARGS[1]) && error("runtime output exists")
open(ARGS[1],"w") do io
    JSON3.write(io,info)
end
