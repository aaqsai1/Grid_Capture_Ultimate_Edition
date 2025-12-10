from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Tuple, Optional
import copy
from enum import Enum

app = FastAPI(title="Grid Capture Game")

# ============= GAME RULES =============
"""
GRID CAPTURE GAME (5x5)

Rules:
1. Two players (Blue=1, Red=2) take turns placing pieces
2. You can only place on empty cells (0)
3. After placing, you "capture" adjacent enemy pieces (up/down/left/right)
4. Captured pieces become yours
5. Game ends when grid is full
6. Winner: Most pieces on the board

Simple but strategic - placement matters!
"""


class Player(int, Enum):
    EMPTY = 0
    BLUE = 1
    RED = 2


class GameState:
    def __init__(self, board=None):
        self.board = board if board else [[0] * 5 for _ in range(5)]

    def copy(self):
        return GameState([row[:] for row in self.board])

    def get_empty_cells(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(5) for c in range(5) if self.board[r][c] == 0]

    def is_valid_move(self, row: int, col: int) -> bool:
        return 0 <= row < 5 and 0 <= col < 5 and self.board[row][col] == 0

    def make_move(self, row: int, col: int, player: int):
        if not self.is_valid_move(row, col):
            return False

        self.board[row][col] = player
        self._capture_adjacent(row, col, player)
        return True

    def _capture_adjacent(self, row: int, col: int, player: int):
        """Capture adjacent enemy pieces"""
        enemy = 3 - player  # If player=1, enemy=2; if player=2, enemy=1
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 5 and 0 <= nc < 5 and self.board[nr][nc] == enemy:
                self.board[nr][nc] = player

    def count_pieces(self) -> Tuple[int, int]:
        blue = sum(row.count(1) for row in self.board)
        red = sum(row.count(2) for row in self.board)
        return blue, red

    def is_game_over(self) -> bool:
        return len(self.get_empty_cells()) == 0

    def get_winner(self) -> Optional[int]:
        if not self.is_game_over():
            return None
        blue, red = self.count_pieces()
        if blue > red:
            return 1
        elif red > blue:
            return 2
        return 0  # Tie


# ============= RULE-BASED AGENT =============
class RuleBasedAgent:
    """Simple rule-based strategy:
    1. Prefer center positions
    2. Prefer positions with more enemy neighbors (more captures)
    3. Prefer corners as secondary choice
    """

    def choose_move(self, state: GameState, player: int) -> Tuple[int, int]:
        empty = state.get_empty_cells()
        if not empty:
            return None

        # Score each position
        scored_moves = []
        for r, c in empty:
            score = 0

            # Rule 1: Center preference (distance from center)
            center_dist = abs(r - 2) + abs(c - 2)
            score += (4 - center_dist) * 2

            # Rule 2: Count capturable enemies
            enemy = 3 - player
            captures = self._count_capturable(state, r, c, enemy)
            score += captures * 10

            # Rule 3: Corner bonus
            if (r, c) in [(0, 0), (0, 4), (4, 0), (4, 4)]:
                score += 3

            scored_moves.append((score, r, c))

        # Pick best move
        scored_moves.sort(reverse=True)
        return (scored_moves[0][1], scored_moves[0][2])

    def _count_capturable(self, state: GameState, row: int, col: int, enemy: int) -> int:
        count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 5 and 0 <= nc < 5 and state.board[nr][nc] == enemy:
                count += 1
        return count


# ============= HEURISTIC SEARCH AGENT =============
class HeuristicSearchAgent:
    """Uses minimax with alpha-beta pruning and heuristic evaluation"""

    def __init__(self, depth=3):
        self.depth = depth

    def choose_move(self, state: GameState, player: int) -> Tuple[int, int]:
        _, best_move = self._minimax(state, self.depth, player, player, float('-inf'), float('inf'))
        return best_move

    def _minimax(self, state: GameState, depth: int, player: int, current_player: int,
                 alpha: float, beta: float) -> Tuple[float, Optional[Tuple[int, int]]]:

        if depth == 0 or state.is_game_over():
            return self._evaluate(state, player), None

        empty = state.get_empty_cells()
        if not empty:
            return self._evaluate(state, player), None

        best_move = None

        if current_player == player:  # Maximizing
            max_eval = float('-inf')
            for r, c in empty:
                new_state = state.copy()
                new_state.make_move(r, c, current_player)
                eval_score, _ = self._minimax(new_state, depth - 1, player, 3 - current_player, alpha, beta)

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = (r, c)

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:  # Minimizing
            min_eval = float('inf')
            for r, c in empty:
                new_state = state.copy()
                new_state.make_move(r, c, current_player)
                eval_score, _ = self._minimax(new_state, depth - 1, player, 3 - current_player, alpha, beta)

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = (r, c)

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def _evaluate(self, state: GameState, player: int) -> float:
        """Heuristic evaluation function"""
        blue, red = state.count_pieces()
        my_pieces = blue if player == 1 else red
        opp_pieces = red if player == 1 else blue

        # Piece advantage
        score = (my_pieces - opp_pieces) * 10

        # Positional bonus (center control)
        for r in range(5):
            for c in range(5):
                if state.board[r][c] == player:
                    center_bonus = 3 - (abs(r - 2) + abs(c - 2))
                    score += center_bonus

        return score


# ============= FASTAPI ENDPOINTS =============

class MoveRequest(BaseModel):
    board: List[List[int]]
    player: int
    agent_type: str  # "rule" or "search"


@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Grid Capture Game - AI Demo</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 900px;
                margin: 40px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255,255,255,0.95);
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                color: #333;
            }
            h1 {
                text-align: center;
                margin-bottom: 10px;
                color: #667eea;
            }
            .rules {
                background: #f0f4ff;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                font-size: 14px;
                line-height: 1.6;
            }
            .game-modes {
                display: flex;
                gap: 15px;
                margin-bottom: 20px;
                justify-content: center;
            }
            .mode-btn {
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                transition: all 0.3s;
            }
            .mode-btn.rule {
                background: #3b82f6;
                color: white;
            }
            .mode-btn.search {
                background: #8b5cf6;
                color: white;
            }
            .mode-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            .grid {
                display: inline-grid;
                grid-template-columns: repeat(5, 70px);
                gap: 5px;
                margin: 20px auto;
                display: block;
                width: fit-content;
            }
            .cell {
                width: 70px;
                height: 70px;
                border: 2px solid #ddd;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 28px;
                cursor: pointer;
                transition: all 0.2s;
                background: white;
            }
            .cell:hover:not(.occupied) {
                background: #f0f4ff;
                border-color: #667eea;
                transform: scale(1.05);
            }
            .cell.blue { background: #3b82f6; color: white; }
            .cell.red { background: #ef4444; color: white; }
            .info {
                text-align: center;
                margin: 20px 0;
                font-size: 18px;
            }
            .scores {
                display: flex;
                justify-content: center;
                gap: 30px;
                margin: 20px 0;
            }
            .score {
                font-size: 20px;
                font-weight: bold;
            }
            .blue-score { color: #3b82f6; }
            .red-score { color: #ef4444; }
            .reset-btn {
                display: block;
                margin: 20px auto;
                padding: 12px 30px;
                background: #10b981;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
            }
            .reset-btn:hover {
                background: #059669;
                transform: translateY(-2px);
            }
            .agent-info {
                background: #fff3cd;
                padding: 10px;
                border-radius: 6px;
                margin: 15px 0;
                text-align: center;
                font-weight: 600;
                color: #856404;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎮 Grid Capture Game - AI Agent Demo</h1>

            <div class="rules">
                <strong>Rules:</strong> Take turns placing pieces. When you place a piece, you capture all adjacent enemy pieces (up/down/left/right). Most pieces wins!
            </div>

            <div class="game-modes">
                <button class="mode-btn rule" onclick="startGame('rule')">
                    🤖 vs Rule-Based Agent
                </button>
                <button class="mode-btn search" onclick="startGame('search')">
                    🧠 vs Heuristic Search Agent
                </button>
            </div>

            <div id="agent-info" class="agent-info" style="display:none;"></div>

            <div class="info" id="info">Select a game mode to start!</div>

            <div class="scores">
                <div class="score blue-score">Blue (You): <span id="blue-score">0</span></div>
                <div class="score red-score">Red (AI): <span id="red-score">0</span></div>
            </div>

            <div class="grid" id="grid"></div>

            <button class="reset-btn" onclick="location.reload()">Reset Game</button>
        </div>

        <script>
            let board = Array(5).fill().map(() => Array(5).fill(0));
            let currentPlayer = 1;
            let gameOver = false;
            let agentType = null;

            function startGame(type) {
                agentType = type;
                const agentName = type === 'rule' ? 'Rule-Based Agent' : 'Heuristic Search Agent';
                const desc = type === 'rule' 
                    ? 'Uses simple rules: prefers center, captures enemies, likes corners'
                    : 'Uses minimax search with lookahead to find optimal moves';
                document.getElementById('agent-info').innerHTML = 
                    `Playing against: <strong>${agentName}</strong><br><small>${desc}</small>`;
                document.getElementById('agent-info').style.display = 'block';
                document.getElementById('info').textContent = 'Your turn! (Blue)';
                renderBoard();
            }

            function renderBoard() {
                const grid = document.getElementById('grid');
                grid.innerHTML = '';

                for (let r = 0; r < 5; r++) {
                    for (let c = 0; c < 5; c++) {
                        const cell = document.createElement('div');
                        cell.className = 'cell';

                        if (board[r][c] === 1) {
                            cell.className += ' blue occupied';
                            cell.textContent = '●';
                        } else if (board[r][c] === 2) {
                            cell.className += ' red occupied';
                            cell.textContent = '●';
                        }

                        if (!gameOver && agentType) {
                            cell.onclick = () => handleClick(r, c);
                        }

                        grid.appendChild(cell);
                    }
                }

                updateScores();
            }

            function updateScores() {
                let blue = 0, red = 0;
                for (let r = 0; r < 5; r++) {
                    for (let c = 0; c < 5; c++) {
                        if (board[r][c] === 1) blue++;
                        if (board[r][c] === 2) red++;
                    }
                }
                document.getElementById('blue-score').textContent = blue;
                document.getElementById('red-score').textContent = red;

                if (blue + red === 25) {
                    gameOver = true;
                    if (blue > red) {
                        document.getElementById('info').textContent = '🎉 You Win!';
                    } else if (red > blue) {
                        document.getElementById('info').textContent = '🤖 AI Wins!';
                    } else {
                        document.getElementById('info').textContent = '🤝 Tie Game!';
                    }
                }
            }

            async function handleClick(r, c) {
                if (board[r][c] !== 0 || currentPlayer !== 1) return;

                // Player move
                await makeMove(r, c, 1);

                if (!gameOver) {
                    // AI move
                    currentPlayer = 2;
                    document.getElementById('info').textContent = 'AI is thinking...';

                    setTimeout(async () => {
                        const response = await fetch('/api/get-move', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                board: board,
                                player: 2,
                                agent_type: agentType
                            })
                        });

                        const data = await response.json();
                        await makeMove(data.move[0], data.move[1], 2);

                        if (!gameOver) {
                            currentPlayer = 1;
                            document.getElementById('info').textContent = 'Your turn!';
                        }
                    }, 500);
                }
            }

            async function makeMove(r, c, player) {
                board[r][c] = player;

                // Capture logic
                const enemy = 3 - player;
                const dirs = [[-1,0], [1,0], [0,-1], [0,1]];

                for (let [dr, dc] of dirs) {
                    const nr = r + dr, nc = c + dc;
                    if (nr >= 0 && nr < 5 && nc >= 0 && nc < 5 && board[nr][nc] === enemy) {
                        board[nr][nc] = player;
                    }
                }

                renderBoard();
            }
        </script>
    </body>
    </html>
    """


@app.post("/api/get-move")
async def get_move(request: MoveRequest):
    """Get AI move based on agent type"""
    try:
        state = GameState(request.board)

        if request.agent_type == "rule":
            agent = RuleBasedAgent()
        elif request.agent_type == "search":
            agent = HeuristicSearchAgent(depth=3)
        else:
            raise HTTPException(400, "Invalid agent type")

        move = agent.choose_move(state, request.player)

        if move is None:
            raise HTTPException(400, "No valid moves available")

        return {"move": move, "agent_type": request.agent_type}

    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)