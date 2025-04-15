import os
import subprocess
import time

def run_command(command):
    """Execute shell command and print output"""
    print(f"Executing: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    # Print output in real-time
    for line in process.stdout:
        print(line.decode().strip())
    
    process.wait()
    
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
        exit(1)

def create_directories():
    """Create project directories if they don't exist"""
    directories = [
        "data/raw",
        "data/processed",
        "notebooks",
        "src/preprocessing",
        "src/modeling",
        "src/evaluation",
        "results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    """Run the entire clickstream analysis pipeline"""
    start_time = time.time()
    
    # 1. Create project structure
    print("Creating project structure...")
    create_directories()
    
    # 2. Generate sample data
    print("\nGenerating sample data...")
    run_command("python src/preprocessing/generate_data.py")
    
    # 3. Data understanding and preprocessing
    print("\nRunning data preprocessing...")
    run_command("python src/preprocessing/preprocess_data.py")
    
    # 4. Feature engineering
    print("\nRunning feature engineering...")
    run_command("python src/preprocessing/feature_engineering.py")
    
    # 5. Apply data mining algorithms
    print("\nRunning data mining algorithms...")
    run_command("python src/modeling/data_mining.py")
    
    # 6. Evaluation and visualization
    print("\nRunning evaluation and visualization...")
    run_command("python src/evaluation/evaluate_results.py")
    
    # 7. Generate final report
    print("\nGenerating final report...")
    run_command("jupyter nbconvert --execute notebooks/final_report.ipynb --to html --output ../results/clickstream_analysis_report.html")
    
    end_time = time.time()
    print(f"\nPipeline completed in {(end_time - start_time)/60:.2f} minutes")
    print("Results available in the 'results' directory")

if __name__ == "__main__":
    main()