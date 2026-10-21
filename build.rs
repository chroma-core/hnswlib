fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Tell cargo to rerun this build script if the bindings change.
    println!("cargo:rerun-if-changed=src/bindings.cpp");
    // Compile the hnswlib bindings.
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file("src/bindings.cpp")
        .define("HAVE_CXX0X", None);

    if build.get_compiler().is_like_msvc() {
        // MSVC-style equivalents of the GCC/Clang flags below. /std:c++14 is the
        // oldest standard MSVC accepts, /fp:fast matches -Ofast's fast-math, and
        // /EHsc is required for the C++ exceptions caught at the FFI boundary.
        build
            .warnings(false)
            .flag("/std:c++14")
            .flag("/O2")
            .flag("/fp:fast")
            .flag("/EHsc")
            .flag("/w");
    } else {
        build
            .flag("-std=c++11")
            .flag("-Ofast")
            .flag("-fPIC")
            .flag("-ftree-vectorize")
            .flag("-w");
    }

    build.compile("bindings");

    Ok(())
}
