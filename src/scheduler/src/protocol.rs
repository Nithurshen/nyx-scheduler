use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[serde(tag = "action", rename_all = "lowercase")]
pub enum InterceptRequest {
    Malloc { bytes: usize },
    Free { ptr: String },
    Compute { grid_x: u32, block_x: u32 },
    // Add the new variant to match the JSON payload
    Cublas_sgemm { m: u64, n: u64, k: u64 }, 
}

#[derive(Debug, Serialize)]
pub struct SchedulerResponse {
    pub status: String,
}