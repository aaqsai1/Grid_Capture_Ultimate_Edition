Grid Capture Ultimate Edition
This project involves designing a high-performance web-based strategy game using the FastAPI framework. It serves as a testbed for Game Theory and Artificial Intelligence, featuring a 5x5 grid where players compete for territory. The project includes sophisticated AI agents utilizing the Minimax algorithm with Alpha-Beta Pruning and Predictive Analytics to challenge human players.
Project Files.

The project consists of several Python files and directories:
•	app.py: Implements the main FastAPI server and game loop. It handles API requests, serves the frontend, and manages the connection between the client and the game logic.
•	templates/index.html: The frontend interface rendered by Jinja2, allowing users to interact with the game via a web browser.

How to Play
1.	Install Dependencies: Before running the project, ensure you have the required libraries installed.
2.	Start the Server: Run app.py to start the Uvicorn server. You will get a link http://0.0.0.0:8000
3.	Access the Game: Open your web browser and navigate to http://0.0.0.0:8000. Change it to http://localhost:8000
4.	Select Mode: Choose between Player vs Player or Player vs AI.
5.	Configure AI: while playing against the computer, select an AI Personality either aggressive or balanced. 
6.	Gameplay: Players take turns placing pieces. Placing a piece adjacent to an opponent's piece captures it. The player with the most pieces when the board is full wins.
AI Agents & Personalities
•	Balanced: Uses Minimax with Alpha-Beta pruning for optimal, long-term strategy.
•	Aggressive: Prioritizes immediate captures and high scores (Greedy search).

Project Objectives
The main objectives of this project are as follows:
•	Implement a high-performance game server using FastAPI.
•	Develop a Minimax AI with Alpha-Beta Pruning capable of strategic depth.
•	Create a Predictive Agent that can forecast opponent threats (Top-N moves).
•	Implement a Power-Up System (Shield, Swap) to introduce non-linear game mechanics.
•	Provide real-time Game Analysis metrics (Momentum, Territory Control) via API endpoints.
Strategy Tips
•	AI Choice: For the most competitive experience, choose the Balanced personality.
•	Power-Ups: The Double power-up is statistically the strongest offensive move. Save it for crowded board states to maximize captures.

# Grid_Capture_Ultimate_Edition
