import subprocess
import sys

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*60}\n")
    
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        print(f"\n[!] Error running {script_name}. Exiting.")
        sys.exit(result.returncode)

if __name__ == "__main__":
    print("STARTING FULL RESEARCH PIPELINE...")
    
    # Step 1: Generate Dataset
    run_script("import_csv.py")
    
    # Step 2: Train CNN Backbone
    run_script("train_cnn.py")
    
    # Step 3: Train LSTM Backbone
    run_script("train_lstm.py")
    
    # Step 4: Walk-Forward Pipeline (Fine-tunes Hybrid Fusion Head on folds)
    run_script("walk_forward_pipeline.py")
    
    print("\n[+] FULL RESEARCH PIPELINE COMPLETED SUCCESSFULLY!")
