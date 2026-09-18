import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from forex_env import ForexHybridEnv
import os

def main():
    print("="*60)
    print("PHASE 4: REINFORCEMENT LEARNING (PPO AGENT)")
    print("="*60)
    
    df = pd.read_csv("dataset_market_cv/metadata.csv")
    df = df.sort_values(by="end_date").reset_index(drop=True)
    
    # We will train the RL agent on the first 60% of data (similar to WFO Fold 1)
    train_size = int(len(df) * 0.6)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"Initializing Environment... (Train Samples: {len(train_df)})")
    
    env = ForexHybridEnv(
        df=train_df, 
        img_dir="dataset_market_cv/images", 
        tensor_dir="dataset_market_cv/tensors", 
        hybrid_model_path="best_model_hybrid_twostep.pth"
    )
    
    # Verify environment
    check_env(env)
    
    # Train Agent
    print("Training PPO Agent. This will take a while as it interacts with the Hybrid Neural Network in real-time...")
    # Because calculating CNN+LSTM features is slow, we use a smaller number of timesteps
    # In a real cluster we'd pre-compute all 128-dim features to memory, but this works as PoC.
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=256)
    
    # Train for 5,000 timesteps as a PoC (In full production, this would be 1M+)
    model.learn(total_timesteps=5000)
    model.save("ppo_forex_agent")
    
    print("\nTraining Complete! Saving Agent.")
    
    # --- TESTING THE AGENT ---
    print("\nRunning Backtest on Out-Of-Sample Data...")
    test_env = ForexHybridEnv(
        df=test_df, 
        img_dir="dataset_market_cv/images", 
        tensor_dir="dataset_market_cv/tensors", 
        hybrid_model_path="best_model_hybrid_twostep.pth"
    )
    
    obs, _ = test_env.reset()
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)
        
    print(f"RL Agent Final Capital (Out-of-Sample): ${test_env.capital:.2f}")

if __name__ == "__main__":
    main()
