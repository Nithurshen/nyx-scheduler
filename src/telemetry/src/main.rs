use nvml_wrapper::Nvml;
use serde::Serialize;
use std::fs;
use std::io::Write;
use std::os::unix::net::UnixListener;
use std::thread;
use std::time::Duration;

const SOCKET_PATH: &str = "/tmp/nyx_telemetry.sock";

// Defines the JSON structure required by Issue #4
#[derive(Serialize)]
struct GpuState {
    gpu_util_pct: u32,
    vram_used_mb: u64,
    vram_total_mb: u64,
    temp_c: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Issue #3: Initialize NVML
    let nvml = Nvml::init()?;
    let device = nvml.device_by_index(0)?;
    println!("Telemetry Daemon Online. Monitoring: {}", device.name()?);

    // Issue #4: Setup UNIX Domain Socket Server
    if fs::metadata(SOCKET_PATH).is_ok() {
        fs::remove_file(SOCKET_PATH)?;
    }

    let listener = UnixListener::bind(SOCKET_PATH)?;
    println!("Broadcasting telemetry on {}", SOCKET_PATH);

    // Listen for incoming connections (e.g., from your Phase 3 Scheduler)
    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                println!("Client connected. Starting telemetry stream...");

                // Spawn a new thread for the connected client to prevent blocking
                thread::spawn(move || {
                    // Re-initialize NVML inside the thread for safety
                    let nvml_thread = Nvml::init().unwrap();
                    let device_thread = nvml_thread.device_by_index(0).unwrap();

                    loop {
                        // Fetch NVML Metrics
                        let util = device_thread.utilization_rates().unwrap();
                        let mem = device_thread.memory_info().unwrap();
                        let temp = device_thread
                            .temperature(
                                nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu,
                            )
                            .unwrap();

                        // Package into struct
                        let state = GpuState {
                            gpu_util_pct: util.gpu,
                            vram_used_mb: mem.used / (1024 * 1024),
                            vram_total_mb: mem.total / (1024 * 1024),
                            temp_c: temp,
                        };

                        // Serialize to JSON and append a newline delimiter
                        let mut json_msg = serde_json::to_string(&state).unwrap();
                        json_msg.push('\n');

                        // Send over the socket; break the loop if the client disconnects
                        if let Err(e) = stream.write_all(json_msg.as_bytes()) {
                            println!("Client disconnected: {}", e);
                            break;
                        }

                        // Configurable polling interval (100ms as per Issue #3)
                        thread::sleep(Duration::from_millis(100));
                    }
                });
            }
            Err(err) => {
                eprintln!("Connection failed: {}", err);
            }
        }
    }

    Ok(())
}
