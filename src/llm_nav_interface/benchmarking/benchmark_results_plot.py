import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob, os

# ---------------- Model rename dictionary ----------------
MODEL_RENAME = {
    "openai/gpt-5": "GPT-5",
    "openai/gpt-5-chat": "GPT-5 Chat",
    "openai/gpt-4o": "GPT-4o",
    "openai/gpt-3.5-turbo": "GPT-3.5",
    "openai/gpt-3.5-turbo-0613": "GPT-3.5-0613",
    "openai/o4-mini": "O4-Mini",
    "anthropic/claude-3.7-sonnet": "Claude-3.7",
    "google/gemini-2.5-pro": "Gemini-2.5",
    "mistralai/mistral-large-2411": "Mistral-Large",
    "mistralai/mistral-7b-instruct": "Mistral-7B",
    "qwen/qwen-2.5-72b-instruct": "Qwen-2.5-72B",
    "meta-llama/llama-3.3-70b-instruct": "LLaMA-3.3-70B",
    "deepseek/deepseek-chat-v3.1": "DeepSeek-Chat-v3.1",
    "deepseek/deepseek-r1": "DeepSeek-R1",
}

# ---------------- Global style ----------------
sns.set_style("whitegrid")
sns.set_context("talk")

# Prepare output folders
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/summaries", exist_ok=True)

# --- Load only raw result CSVs (skip summary_*.csv) ---
all_files = [
    f for f in glob.glob(os.path.join("results", "**", "*.csv"), recursive=True)
    if not os.path.basename(f).startswith("summary_")
]

summaries = []
df_list = []

# ---------------- Per-file summaries & plots ----------------
for f in all_files:
    df = pd.read_csv(f, sep=",")
    df_list.append(df)

    summary = df.groupby("model").agg(
        avg_latency=("latency_s", "mean"),
        std_latency=("latency_s", "std"),
        total_calls=("command", "count"),
        success_rate=("parsed_ok", "mean"),
        avg_tokens_in=("tokens_in", "mean"),
        avg_tokens_out=("tokens_out", "mean"),
        avg_tokens_total=("tokens_total", "mean")
    ).reset_index()

    summary["success_rate"] = summary["success_rate"] * 100
    summary["file"] = os.path.basename(f)
    summaries.append(summary)

    # Save summary CSV (with full model names)
    summary.to_csv(f"results/summaries/summary_{os.path.basename(f)}", index=False)

    # Temporary column just for plotting
    plot_data = summary.copy()
    plot_data["Models"] = plot_data["model"].map(MODEL_RENAME).fillna(plot_data["model"])

    # --- Latency plot ---
    plt.figure(figsize=(12,6))
    sns.barplot(data=plot_data, x="Models", y="avg_latency",
                hue="Models", dodge=False, legend=False,
                palette="muted", errorbar=None)
    plt.errorbar(x=range(len(plot_data["Models"])),
                 y=plot_data["avg_latency"],
                 yerr=plot_data["std_latency"],
                 fmt='none', c='black', capsize=5)
    plt.title(f"Average Latency per Model ({os.path.basename(f)})")
    plt.ylabel("Latency (s)")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"results/plots/latency_{os.path.basename(f)}.png", dpi=300)
    plt.show()

    # --- Success Rate plot ---
    plt.figure(figsize=(12,6))
    sns.barplot(data=plot_data, x="Models", y="success_rate",
                hue="Models", dodge=False, legend=False,
                palette="deep", errorbar=None)
    plt.title(f"Success Rate per Model ({os.path.basename(f)})")
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"results/plots/success_{os.path.basename(f)}.png", dpi=300)
    plt.show()

    # --- Per-file Scatter plot (per-call latency) ---
    df_plot = df.copy()
    df_plot["Models"] = df_plot["model"].map(MODEL_RENAME).fillna(df_plot["model"])

    plt.figure(figsize=(12,6))
    sns.stripplot(data=df_plot, x="Models", y="latency_s",
                  hue="Models", dodge=False, legend=False,
                  jitter=True, alpha=0.6, palette="muted", size=4)
    plt.title(f"Per-call Latency Scatter ({os.path.basename(f)})")
    plt.ylabel("Latency (s)")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"results/plots/latency_scatter_{os.path.basename(f)}.png", dpi=300)
    plt.show()


# ---------------- Combined summary ----------------
df_all = pd.concat(df_list, ignore_index=True)
summary_all = df_all.groupby("model").agg(
    avg_latency=("latency_s", "mean"),
    std_latency=("latency_s", "std"),
    total_calls=("command", "count"),
    success_rate=("parsed_ok", "mean"),
    avg_tokens_in=("tokens_in", "mean"),
    avg_tokens_out=("tokens_out", "mean"),
    avg_tokens_total=("tokens_total", "mean")
).reset_index()

summary_all["success_rate"] = summary_all["success_rate"] * 100

# Save combined summary with full names
summary_all.to_csv("results/summaries/summary_combined.csv", index=False)

# Temporary column just for plotting
plot_all = summary_all.copy()
plot_all["Models"] = plot_all["model"].map(MODEL_RENAME).fillna(plot_all["model"])

print("\n=== Combined Summary Across All Files ===")
display(summary_all.round(3))

# --- Combined Latency plot ---
plt.figure(figsize=(12,6))
sns.barplot(data=plot_all, x="Models", y="avg_latency",
            hue="Models", dodge=False, legend=False,
            palette="muted", errorbar=None)
plt.errorbar(x=range(len(plot_all["Models"])),
             y=plot_all["avg_latency"],
             yerr=plot_all["std_latency"],
             fmt='none', c='black', capsize=5)
plt.title("Average Latency per Model (All Runs)")
plt.ylabel("Latency (s)")
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig("results/plots/latency_combined.png", dpi=300)
plt.show()

# --- Combined Success Rate plot ---
plt.figure(figsize=(12,6))
sns.barplot(data=plot_all, x="Models", y="success_rate",
            hue="Models", dodge=False, legend=False,
            palette="deep", errorbar=None)
plt.title("Parsing Success Rate per Model (All Runs)")
plt.ylabel("Success Rate (%)")
plt.ylim(0, 100)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig("results/plots/success_combined.png", dpi=300)
plt.show()

# --- Combined Tokens plot ---
plt.figure(figsize=(12,6))
sns.barplot(data=plot_all, x="Models", y="avg_tokens_total",
            hue="Models", dodge=False, legend=False,
            palette="pastel", errorbar=None)
plt.title("Average Tokens per Call (All Runs)")
plt.ylabel("Tokens")
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig("results/plots/tokens_combined.png", dpi=300)
plt.show()

# --- Combined Scatter plot (per-call latency) ---
df_all["Models"] = df_all["model"].map(MODEL_RENAME).fillna(df_all["model"])
df_all_plot = df_all.copy()
df_all_plot["Models"] = df_all_plot["model"].map(MODEL_RENAME).fillna(df_all_plot["model"])

plt.figure(figsize=(12,6))
sns.stripplot(data=df_all_plot, x="Models", y="latency_s",
              hue="Models", dodge=False, legend=False,
              jitter=True, alpha=0.6, palette="muted", size=4)
plt.title("Per-call Latency Scatter (All Runs)")
plt.ylabel("Latency (s)")
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig("results/plots/latency_scatter_combined.png", dpi=300)
plt.show()
