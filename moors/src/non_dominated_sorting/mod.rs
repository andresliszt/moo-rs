mod fds;

pub(crate) use fds::build_fronts;
pub use fds::{dominates, dominates_weak, fast_non_dominated_sorting};
