fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Tell cargo to rerun this build script if the bindings or the headers they
    // include change. Without the header directory, edits to hnswlib/*.h are linked
    // against a stale object.
    println!("cargo:rerun-if-changed=src/bindings.cpp");
    println!("cargo:rerun-if-changed=hnswlib");
    // Compile the hnswlib bindings.
    cc::Build::new()
        .cpp(true)
        .file("src/bindings.cpp")
        .flag("-std=c++11")
        .flag("-Ofast")
        .flag("-DHAVE_CXX0X")
        .flag("-fPIC")
        .flag("-ftree-vectorize")
        .flag("-w")
        .compile("bindings");

    Ok(())
}
