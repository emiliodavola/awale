#!/usr/bin/env julia
"""
    verify_doc_coverage.jl

Walk all .jl source files and report docstring coverage.
Exits 0 if every file has at least as many docstrings as documented entities, 1 otherwise.
"""

function find_source_files()::Vector{String}
    files = String[]

    # Walk src/ recursively
    for (root, _, filenames) in walkdir(joinpath(@__DIR__, "..", "src"))
        for f in filenames
            endswith(f, ".jl") && push!(files, joinpath(root, f))
        end
    end

    # Walk root directory for top-level .jl files (train.jl, play.jl, etc.)
    root_dir = abspath(joinpath(@__DIR__, ".."))
    for f in readdir(root_dir)
        path = joinpath(root_dir, f)
        isfile(path) && endswith(f, ".jl") && push!(files, path)
    end

    # Walk scripts/ directory
    scripts_dir = @__DIR__
    for f in readdir(scripts_dir)
        path = joinpath(scripts_dir, f)
        isfile(path) && endswith(f, ".jl") && f != basename(@__FILE__) && push!(files, path)
    end

    return sort!(unique!(files))
end

function count_docstrings(text::String)::Int
    count = 0
    in_docstring = false
    for line in eachsplit(text, '\n')
        stripped = strip(line)
        if startswith(stripped, "\"\"\"")
            if in_docstring
                in_docstring = false
                count += 1
            else
                in_docstring = true
            end
        end
    end
    return count
end

function count_definitions(text::String)::Int
    count = 0
    for line in eachsplit(text, '\n')
        stripped = strip(line)
        # Count top-level definitions: function, struct, mutable struct, macro
        if startswith(stripped, "function ") ||
           startswith(stripped, "struct ") ||
           startswith(stripped, "mutable struct ") ||
           startswith(stripped, "macro ")
            count += 1
        end
    end
    return count
end

function check_file(filepath::String)::Tuple{Bool, Int, Int, String}
    text = read(filepath, String)
    ndoc = count_docstrings(text)
    ndef = count_definitions(text)

    rp = Base.relpath(filepath, abspath(joinpath(@__DIR__, "..")))
    status = ndoc >= ndef ? "PASS" : "FAIL"
    ok = ndoc >= ndef
    return ok, ndoc, ndef, rp
end

function main()
    files = find_source_files()
    all_pass = true
    results = []

    println("=" ^ 64)
    println("Docstring Coverage Report")
    println("=" ^ 64)
    println()

    for filepath in files
        ok, ndoc, ndef, rp = check_file(filepath)
        push!(results, (ok, ndoc, ndef, rp))
        if !ok
            all_pass = false
        end
    end

    # Print summary table
    header = "  | Status | Docstrings | Definitions | File"
    println(header)
    println("-" ^ length(header))
    for (ok, ndoc, ndef, rp) in results
        status_str = ok ? "PASS" : "FAIL"
        println("  | $status_str | $(lpad(ndoc, 10)) | $(lpad(ndef, 11)) | $rp")
    end

    println()
    println("-" ^ length(header))
    total_doc = sum(r[2] for r in results)
    total_def = sum(r[3] for r in results)
    println("  | Total  | $(lpad(total_doc, 10)) | $(lpad(total_def, 11)) | $(length(results)) files")
    println()

    if all_pass
        println("✓ All files have sufficient docstring coverage.")
        return true
    else
        println("✗ Some files have more definitions than docstrings. Review FAIL entries above.")
        return false
    end
end

if basename(PROGRAM_FILE) == basename(@__FILE__)
    success = main()
    exit(success ? 0 : 1)
end
