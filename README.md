# ros2_capstone_SPI
LLM-Guided Robot Navigation and Benchmarking in ROS2

This repository contains the implementation of LLM-Guided Robot Navigation and Benchmarking, a research project exploring natural-language robot control using multiple Large Language Models (LLMs) integrated with ROS 2 Humble, Nav2, and Ignition Gazebo Fortress.

The system converts free-form natural language commands into navigation goals, and benchmarks end-to-end performance across:

Multiple LLM families (GPT, Claude, Gemini, Mistral, DeepSeek, LLaMA/Groq)

Multiple Nav2 planners: DWB, TEB, RPP

Latency, parsing accuracy, path efficiency, and task success rate

This is one of the first reproducible multi-LLM + multi-planner benchmarks in a ROS 2 navigation pipeline using a TurtleBot4 simulation.

📌 Repository Structure
```bash
├── src/
│   ├── llm_interface/
│   ├── nav2_planner_profiles/
│   ├── robot_controller/
│   └── data_visualisation/
├── results/
│   ├── latency/
│   ├── success/
│   ├── tokens/
│   └── planner_comparison/
├── docs/
│   ├── project_proposal.pdf
│   ├── literature_review.pdf
│   └── sensors_paper_draft.pdf
└── README.md
```


```bash
🧠 System Architecture
Natural Language Input
        ↓
LLM API (OpenRouter / Groq / Official APIs)
        ↓
Structured JSON Output → Goal Pose
        ↓
ROS2 Node (goal publisher)
        ↓
Nav2 Local Planner (DWB / TEB / RPP)
        ↓
TurtleBot4 Navigation in Gazebo
```


🔧 Installation
1. Install ROS 2 Humble
sudo apt install ros-humble-desktop-full

2. Install TurtleBot4 + Nav2 + Gazebo Fortress
sudo apt install ros-humble-turtlebot4*
sudo apt install ros-humble-nav2*
sudo apt install ros-humble-gazebo-ros-pkgs

3. Python dependencies
pip install openai groq pyyaml pandas seaborn matplotlib

4. Set environment keys

Add to .bashrc:

export OPENROUTER_API_KEY=your_key
export GROQ_API_KEY=your_key

▶️ Running the System
1. Launch TurtleBot4 simulation
ros2 launch turtlebot4_gazebo turtlebot4_world.launch.py

2. Start Nav2
ros2 launch nav2_bringup navigation_launch.py

3. Run the LLM Benchmark
python3 src/llm_interface/benchmark_openrouter.py

4. Send a natural-language command
ros2 topic pub /nl_command std_msgs/String "Go to the table near the window"

📊 Benchmarking Metrics
✔ LLM Latency

API response time

Parsing time

Navigation execution time

✔ Parsing Accuracy

JSON structured response validation:

{"goal": {"x": 1.2, "y": -3.4}, "planner": "TEB"}

✔ Planner Evaluation

Path length

Time-to-goal

Oscillations / recovery behaviour

Success rate

✔ Token & Cost Tracking

Automatic logging for OpenRouter models.

📁 Reproducibility

Each run generates a timestamped folder:

results/YYYY-MM-DD_HH-MM-SS/


Containing:

Raw CSV logs

Summary tables

Latency plots

Success plots

Token usage reports

Planner comparison metrics

🤝 Contact

Author: Murat Das
Email: dasmuratr@gmail.com

Supervisor: Dr. Zawar Hussain
Sydney Polytechnic Institute – Data Science & AI Faculty
