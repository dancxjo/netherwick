include!("surface/extractor.rs");
include!("surface/projection.rs");
include!("surface/planes.rs");
include!("surface/obstacles.rs");

#[cfg(test)]
#[path = "surface_tests.rs"]
mod tests;
