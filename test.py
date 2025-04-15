import subprocess
import argparse
import statistics
import time

def run_binary(binary_path, args, num_runs):
    """
    Run the binary with the given arguments for a specified number of times
    and return all execution times.
    """
    times = []
    
    for i in range(num_runs):
        # Construct the command
        cmd = [binary_path] + args
        
        # Run the command and capture output
        start_time = time.time()  # Backup timer in case binary doesn't report its own time
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout.strip()
            
            # Try to parse the time from the binary's output
            try:
                execution_time = float(output)
                times.append(execution_time)
            except ValueError:
                # If binary output isn't a float time, use our measured time
                execution_time = time.time() - start_time
                times.append(execution_time)
                print(f"Warning: Could not parse time from binary output. Using measured time: {execution_time:.6f} seconds")
                
        except subprocess.CalledProcessError as e:
            print(f"Error running binary (run {i+1}/{num_runs}): {e}")
            print(f"Error output: {e.stderr}")
            continue
            
        # print(f"Run {i+1}/{num_runs}: {times[-1]:.6f} seconds")
    
    return times

def main():
    parser = argparse.ArgumentParser(description="Run a C++ binary multiple times and calculate average execution time")
    
    parser.add_argument("binary", help="Path to the binary to execute")
    parser.add_argument("--runs", type=int, default=10, help="Number of times to run the binary (default: 5)")
    
    # Arguments for the binary
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--sparsity", type=float, required=True)
    
    args = parser.parse_args()
    
    # Convert arguments to list for binary
    binary_args = [
        str(args.M),
        str(args.K),
        str(args.N),
        str(args.sparsity),
    ]
    
    # Run the binary and collect times
    times = run_binary(args.binary, binary_args, args.runs)
    
    if not times:
        print("No successful runs recorded.")
        return
    
    # Calculate statistics
    avg_time = statistics.mean(times)
    
    print(f"{avg_time:.6f}")

if __name__ == "__main__":
    main()
