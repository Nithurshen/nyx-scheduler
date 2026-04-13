use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};
use std::fs;

mod protocol;
use protocol::{InterceptRequest, SchedulerResponse};

const SOCK_PATH: &str = "/tmp/nyx.sock";
const TELEMETRY_SOCK_PATH: &str = "/tmp/nyx_telemetry.sock";

#[derive(Debug)]
enum JobClass {
    Lightweight,
    ComputeBound,
    MemoryBound,
}

// 1. The Profiler: Analyzes the computational graph's shape
fn classify_job(m: u64, n: u64, k: u64, requested_bytes: u64) -> JobClass {
    // Estimated floating point operations for a matrix multiplication
    let estimated_flops = 2 * m * n * k;
    
    // How math-heavy is this job relative to its memory footprint?
    let compute_density = estimated_flops as f64 / (requested_bytes as f64 + 1.0);

    // Thresholds: Can be tweaked based on your specific GPUs
    if estimated_flops < 500_000_000 && requested_bytes < 100_000_000 {
        JobClass::Lightweight
    } else if compute_density > 50.0 {
        JobClass::ComputeBound
    } else {
        JobClass::MemoryBound
    }
}

// 1. Internal State Structure
#[derive(Debug, Default)]
struct SchedulerState {
    pub vram_total_mb: u64,
    pub vram_used_mb: u64,
    pub vram_reserved_bytes: u64, 
    pub gpu_util_pct: u32, // NEW: Track GPU compute load for time-slicing
}
// Struct matching the JSON emitted by the telemetry daemon
#[derive(Debug, Deserialize)]
struct GpuState {
    gpu_util_pct: u32,
    vram_used_mb: u64,
    vram_total_mb: u64,
    temp_c: u32,
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    // Wrap our state in an Arc<Mutex<>> so ownership can be safely shared 
    // across the telemetry listener and all client request tasks.
    let state = Arc::new(Mutex::new(SchedulerState::default()));

    // 2. Spawn the Telemetry Subscriber Task
    let state_clone = Arc::clone(&state);
    tokio::spawn(async move {
        subscribe_to_telemetry(state_clone).await;
    });

    // Start the Interceptor Listener
    if fs::metadata(SOCK_PATH).is_ok() {
        fs::remove_file(SOCK_PATH)?;
    }
    let listener = UnixListener::bind(SOCK_PATH)?;
    println!("Nyx Scheduler listening on {}...", SOCK_PATH);

    loop {
        match listener.accept().await {
            Ok((stream, _addr)) => {
                let task_state = Arc::clone(&state);
                // Spawn a task for each PyTorch DL job
                tokio::spawn(async move {
                    if let Err(e) = handle_client(stream, task_state).await {
                        eprintln!("Error handling client: {}", e);
                    }
                });
            }
            Err(e) => eprintln!("Failed to accept connection: {}", e),
        }
    }
}

// Connects to Nitin's NVML stream and updates the internal state continuously
async fn subscribe_to_telemetry(state: Arc<Mutex<SchedulerState>>) {
    loop {
        match UnixStream::connect(TELEMETRY_SOCK_PATH).await {
            Ok(stream) => {
                println!("Connected to Telemetry Daemon.");
                let mut reader = BufReader::new(stream);
                let mut line = String::new();

                while let Ok(bytes_read) = reader.read_line(&mut line).await {
                    if bytes_read == 0 { break; } // Stream closed

                    if let Ok(gpu_state) = serde_json::from_str::<GpuState>(&line) {
                        let mut s = state.lock().await;
                        s.vram_total_mb = gpu_state.vram_total_mb;
                        s.vram_used_mb = gpu_state.vram_used_mb;
                        s.gpu_util_pct = gpu_state.gpu_util_pct; // NEW: Update load
                        s.vram_reserved_bytes = 0; 
                    }
                    line.clear();
                }
            }
            Err(_) => {
                // If telemetry isn't up yet, wait and retry
                sleep(Duration::from_secs(2)).await;
            }
        }
    }
}

// 3. The Hold/Release Queuing Logic
async fn handle_client(mut stream: UnixStream, state: Arc<Mutex<SchedulerState>>) -> std::io::Result<()> {
    let mut buffer = [0; 1024];

    loop {
        let n = stream.read(&mut buffer).await?;
        if n == 0 { break; }

        let msg_str = String::from_utf8_lossy(&buffer[..n]);
        
        match serde_json::from_str::<InterceptRequest>(&msg_str) {
            Ok(InterceptRequest::Malloc { bytes }) => {
                println!("Client requesting {} bytes...", bytes);
                
                // Queuing logic: Loop and yield until sufficient VRAM is available
                loop {
                    let mut s = state.lock().await;
                    let vram_used_bytes = (s.vram_used_mb * 1024 * 1024) + s.vram_reserved_bytes;
                    let vram_total_bytes = s.vram_total_mb * 1024 * 1024;
                    
                    // Safety margin: ensure we have the bytes available
                    if vram_used_bytes + (bytes as u64) < vram_total_bytes {
                        // We have space! Reserve it internally so we don't double-allocate 
                        // before the next telemetry tick.
                        s.vram_reserved_bytes += bytes as u64;
                        println!("Granted {} bytes. Total reserved: {}", bytes, s.vram_reserved_bytes);
                        break; 
                    }
                    
                    // Explicitly drop the Mutex lock so other tasks can update the state while we wait
                    drop(s);
                    
                    // "Wait" by sleeping the task, holding the CUDA call intercepted
                    sleep(Duration::from_millis(50)).await;
                }

                // Release the CUDA call
                let response = SchedulerResponse { status: "Go".to_string() };
                stream.write_all(serde_json::to_string(&response).unwrap().as_bytes()).await?;
            }
            Ok(InterceptRequest::Free { ptr }) => {
                println!("Client freeing ptr: {}", ptr);
                // We let the NVML telemetry physically catch the drop, but we acknowledge it
                let response = SchedulerResponse { status: "Go".to_string() };
                stream.write_all(serde_json::to_string(&response).unwrap().as_bytes()).await?;
            }
            Ok(InterceptRequest::Cublas_sgemm { m, n, k }) => {
    println!("Intercepted Graph Shape: m={}, n={}, k={}", m, n, k);
    
    // We assume an average of 10MB temporary workspace for standard GEMM if not explicitly tracked
    let estimated_vram = 10_485_760; 
    let job_class = classify_job(m, n, k, estimated_vram);
    
    println!("Job Classified as: {:?}", job_class);

    loop {
        let mut s = state.lock().await;
        let mut should_grant = false;

        // 2. The Decision Matrix: Route jobs based on hardware state
        match job_class {
            JobClass::Lightweight => {
                // Squeeze small jobs in unless the GPU is absolutely melting
                if s.gpu_util_pct < 95 { should_grant = true; }
            }
            JobClass::ComputeBound => {
                // If the GPU is already crunching heavy math, QUEUE this job to prevent interference
                if s.gpu_util_pct < 75 { should_grant = true; }
            }
            JobClass::MemoryBound => {
                // If it's memory heavy, we care more about VRAM availability than SM compute load
                let vram_used_bytes = (s.vram_used_mb * 1024 * 1024) + s.vram_reserved_bytes;
                if vram_used_bytes + estimated_vram < (s.vram_total_mb * 1024 * 1024) {
                    should_grant = true;
                    // 2. ADD THIS LINE: Actually reserve the memory!
                    s.vram_reserved_bytes += estimated_vram; 
                }
            }
        }

        if should_grant {
            break; // Send the "Go" signal to the C++ Interceptor
        }

        // Job queued. Drop lock and yield to software queue.
        drop(s);
        tokio::time::sleep(tokio::time::Duration::from_millis(15)).await;
    }

    let response = SchedulerResponse { status: "Go".to_string() };
    stream.write_all(serde_json::to_string(&response).unwrap().as_bytes()).await?;
}
            // NEW: Handle compute requests
            Ok(InterceptRequest::Compute { grid_x, block_x }) => {
                println!("Compute slice requested (Grid X: {}, Block X: {})", grid_x, block_x);
                
                // Queuing logic: Loop and yield if the GPU is currently saturated
                loop {
                    let s = state.lock().await;
                    
                    // Time-Slicing Threshold: If GPU is above 90% utilization, 
                    // we hold this kernel launch to let current SMs flush out.
                    if s.gpu_util_pct < 90 {
                        break; 
                    }
                    
                    // Drop lock and sleep briefly to act as a software queue
                    drop(s);
                    sleep(Duration::from_millis(10)).await;
                }
                
                let response = SchedulerResponse { status: "Go".to_string() };
                stream.write_all(serde_json::to_string(&response).unwrap().as_bytes()).await?;
            }
            Err(e) => eprintln!("Failed to parse message: {}", e),
        }
    }
    Ok(())
}