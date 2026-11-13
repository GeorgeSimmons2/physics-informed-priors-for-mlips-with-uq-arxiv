# Where things live.  A stage writes under results/ (or $OUT) and reads its inputs with `input`, which falls back to
# published/: every stage can therefore be run on its own, starting from the published results of the stages before.

"The repository root."
root() = pkgdir(@__MODULE__)

"A path under the results directory (`\$OUT`, default `results/`); its parent directory is created."
function results(parts...)
    path = joinpath(get(ENV, "OUT", joinpath(root(), "results")), parts...)
    mkpath(dirname(path))
    return path
end

"A path under `published/`, the results the paper's figures were drawn from."
published(parts...) = joinpath(root(), "published", parts...)

"An input file: this run's result if it exists, the published one otherwise."
function input(parts...)
    for path in (joinpath(get(ENV, "OUT", joinpath(root(), "results")), parts...), published(parts...))
        isfile(path) && return path
    end
    error("no $(joinpath(parts...)) under results/ or published/: run the stage that writes it first")
end
