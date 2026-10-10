using Pkg
using TOML
pins = TOML.parsefile(ARGS[1])
Pkg.add([PackageSpec(name="Oceananigans", version=pins["oceananigans"]),
         PackageSpec(name="CUDA", version=pins["cuda"]),
         PackageSpec(name="GPUArrays"), PackageSpec(name="NCDatasets"), PackageSpec(name="JSON3")])
Pkg.pin([PackageSpec(name="Oceananigans"), PackageSpec(name="CUDA")])
@assert Pkg.dependencies()[Base.UUID("9e8cae18-63c1-5223-a75c-80ca9d6e9a09")].version == VersionNumber(pins["oceananigans"])
Pkg.status()
