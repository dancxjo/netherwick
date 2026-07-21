include!("beliefs/types.rs");
include!("beliefs/updater.rs");
include!("beliefs/helpers.rs");

#[cfg(test)]
#[path = "beliefs_tests.rs"]
mod tests;
