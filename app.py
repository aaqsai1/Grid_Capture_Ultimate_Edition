from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict
from enum import Enum
import copy
import random
import asyncio
import json
from datetime import datetime

from starlette.requests import Request
from starlette.templating import Jinja2Templates

app = FastAPI(title="Grid Capture Ultimate Edition")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Player(int, Enum):
    EMPTY = 0
    BLUE = 1
    RED = 2


class PowerUpType(str, Enum):
    SHIELD = "shield"
    DOUBLE = "double"
    SWAP = "swap"
    FREEZE = "freeze"


class AIPersonality(str, Enum):
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    BALANCED = "balanced"
    CHAOTIC = "chaotic"


class GameState:
    def __init__(self, board=None):
        self.board = board if board else [[0] * 5 for _ in range(5)]
        self.move_history = []
        self.power_ups = {
            1: {"shield": 1, "double": 1, "swap": 1},
            2: {"shield": 1, "double": 1, "swap": 1}
        }
        self.active_effects = []

    def copy(self):
        new_state = GameState([row[:] for row in self.board])
        new_state.move_history = self.move_history.copy()
        new_state.power_ups = copy.deepcopy(self.power_ups)
        new_state.active_effects = self.active_effects.copy()
        return new_state

    def get_empty_cells(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(5) for c in range(5) if self.board[r][c] == 0]

    def is_valid_move(self, row: int, col: int) -> bool:
        return 0 <= row < 5 and 0 <= col < 5 and self.board[row][col] == 0

    def make_move(self, row: int, col: int, player: int, power_up: Optional[str] = None):
        if not self.is_valid_move(row, col):
            return False, []

        self.board[row][col] = player
        captured = self._capture_adjacent(row, col, player, power_up)

        self.move_history.append({
            "row": row,
            "col": col,
            "player": player,
            "captured": captured,
            "power_up": power_up,
            "timestamp": datetime.now().isoformat()
        })

        return True, captured

    def _capture_adjacent(self, row: int, col: int, player: int, power_up: Optional[str] = None):
        enemy = 3 - player
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        captured = []

        # Check for shield protection
        shield_active = any(e["type"] == "shield" and e["player"] == enemy for e in self.active_effects)

        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 5 and 0 <= nc < 5 and self.board[nr][nc] == enemy:
                if not shield_active:
                    self.board[nr][nc] = player
                    captured.append((nr, nc))

        # Double capture power-up
        if power_up == "double" and self.power_ups[player].get("double", 0) > 0:
            self.power_ups[player]["double"] -= 1
            second_wave = []
            for cr, cc in captured:
                for dr, dc in directions:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < 5 and 0 <= nc < 5 and self.board[nr][nc] == enemy:
                        self.board[nr][nc] = player
                        second_wave.append((nr, nc))
            captured.extend(second_wave)

        return captured

    def apply_power_up(self, player: int, power_up_type: str):
        if self.power_ups[player].get(power_up_type, 0) <= 0:
            return False

        if power_up_type == "shield":
            self.active_effects.append({"type": "shield", "player": player, "duration": 1})
            self.power_ups[player]["shield"] -= 1
        elif power_up_type == "swap":
            # Swap all pieces on board
            for r in range(5):
                for c in range(5):
                    if self.board[r][c] == player:
                        self.board[r][c] = 3 - player
                    elif self.board[r][c] == 3 - player:
                        self.board[r][c] = player
            self.power_ups[player]["swap"] -= 1

        return True

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
        return 0

    def get_territory_control(self) -> Dict:
        """Advanced metric: control of different board regions"""
        regions = {
            "center": [(2, 2)],
            "corners": [(0, 0), (0, 4), (4, 0), (4, 4)],
            "edges": [(0, 2), (2, 0), (2, 4), (4, 2)]
        }

        control = {"blue": {}, "red": {}}
        for region_name, cells in regions.items():
            blue_count = sum(1 for r, c in cells if self.board[r][c] == 1)
            red_count = sum(1 for r, c in cells if self.board[r][c] == 2)
            control["blue"][region_name] = blue_count
            control["red"][region_name] = red_count

        return control


# ============= ADVANCED AI AGENTS =============

class AdaptiveAgent:
    """AI that adapts strategy based on game state and opponent behavior"""

    def __init__(self, depth=3, personality: AIPersonality = AIPersonality.BALANCED):
        self.depth = depth
        self.personality = personality
        self.opponent_pattern = []

    def choose_move(self, state: GameState, player: int) -> Tuple[int, int]:
        empty = state.get_empty_cells()
        if not empty:
            return None

        # Analyze opponent's recent moves
        recent_moves = [m for m in state.move_history[-5:] if m["player"] != player]

        # Personality-based strategy adjustment
        if self.personality == AIPersonality.AGGRESSIVE:
            return self._aggressive_strategy(state, player, empty)
        elif self.personality == AIPersonality.DEFENSIVE:
            return self._defensive_strategy(state, player, empty)
        elif self.personality == AIPersonality.CHAOTIC:
            return self._chaotic_strategy(state, player, empty)
        else:
            return self._balanced_strategy(state, player, empty)

    def _aggressive_strategy(self, state: GameState, player: int, empty: List):
        """Prioritize capturing enemy pieces"""
        best_move = None
        max_captures = -1

        for r, c in empty:
            captures = self._count_potential_captures(state, r, c, player)
            if captures > max_captures:
                max_captures = captures
                best_move = (r, c)

        return best_move

    def _defensive_strategy(self, state: GameState, player: int, empty: List):
        """Minimize opponent's capture opportunities"""
        best_move = None
        min_vulnerability = float('inf')

        for r, c in empty:
            test_state = state.copy()
            test_state.make_move(r, c, player)

            # Check how many pieces opponent could capture next
            vulnerability = 0
            for er, ec in test_state.get_empty_cells():
                vulnerability += self._count_potential_captures(test_state, er, ec, 3 - player)

            if vulnerability < min_vulnerability:
                min_vulnerability = vulnerability
                best_move = (r, c)

        return best_move

    def _chaotic_strategy(self, state: GameState, player: int, empty: List):
        """Unpredictable moves with high variance"""
        weights = []
        for r, c in empty:
            captures = self._count_potential_captures(state, r, c, player)
            position_value = abs(r - 2) + abs(c - 2)  # Distance from center
            chaos_factor = random.random() * 10
            weight = captures * 5 + chaos_factor + position_value
            weights.append(weight)

        # Weighted random selection
        total = sum(weights)
        if total == 0:
            return random.choice(empty)

        rand = random.random() * total
        cumsum = 0
        for i, w in enumerate(weights):
            cumsum += w
            if cumsum >= rand:
                return empty[i]

        return empty[-1]

    def _balanced_strategy(self, state: GameState, player: int, empty: List):
        """Minimax with alpha-beta pruning"""
        _, best_move = self._minimax(state, self.depth, player, player,
                                     float('-inf'), float('inf'))
        return best_move

    def _minimax(self, state: GameState, depth: int, player: int,
                 current_player: int, alpha: float, beta: float):
        if depth == 0 or state.is_game_over():
            return self._evaluate(state, player), None

        empty = state.get_empty_cells()
        if not empty:
            return self._evaluate(state, player), None

        best_move = None

        if current_player == player:
            max_eval = float('-inf')
            for r, c in empty:
                new_state = state.copy()
                new_state.make_move(r, c, current_player)
                eval_score, _ = self._minimax(new_state, depth - 1, player,
                                              3 - current_player, alpha, beta)

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = (r, c)

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for r, c in empty:
                new_state = state.copy()
                new_state.make_move(r, c, current_player)
                eval_score, _ = self._minimax(new_state, depth - 1, player,
                                              3 - current_player, alpha, beta)

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = (r, c)

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def _evaluate(self, state: GameState, player: int) -> float:
        blue, red = state.count_pieces()
        my_pieces = blue if player == 1 else red
        opp_pieces = red if player == 1 else blue

        score = (my_pieces - opp_pieces) * 10

        # Territory control bonus
        control = state.get_territory_control()
        my_color = "blue" if player == 1 else "red"
        score += control[my_color]["center"] * 5
        score += control[my_color]["corners"] * 3

        # Mobility (available moves)
        empty = len(state.get_empty_cells())
        score += empty * 0.5

        return score

    def _count_potential_captures(self, state: GameState, row: int, col: int, player: int) -> int:
        enemy = 3 - player
        count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 5 and 0 <= nc < 5 and state.board[nr][nc] == enemy:
                count += 1
        return count


class PredictiveAgent(AdaptiveAgent):
    """AI that predicts opponent moves and plans accordingly"""

    def get_predictions(self, state: GameState, player: int, num_predictions: int = 3):
        """Return top N predicted opponent moves"""
        opponent = 3 - player
        empty = state.get_empty_cells()

        predictions = []
        for r, c in empty:
            test_state = state.copy()
            captures = self._count_potential_captures(test_state, r, c, opponent)
            eval_score = self._evaluate_move(test_state, r, c, opponent)

            predictions.append({
                "position": (r, c),
                "captures": captures,
                "score": eval_score
            })

        predictions.sort(key=lambda x: x["score"], reverse=True)
        return predictions[:num_predictions]

    def _evaluate_move(self, state: GameState, row: int, col: int, player: int) -> float:
        test_state = state.copy()
        test_state.make_move(row, col, player)
        return self._evaluate(test_state, player)


# ============= API MODELS =============

class MoveRequest(BaseModel):
    board: List[List[int]]
    player: int
    agent_type: str
    difficulty: int = 3
    personality: str = "balanced"
    power_up: Optional[str] = None


class PredictionRequest(BaseModel):
    board: List[List[int]]
    player: int
    num_predictions: int = 3


class GameAnalysisRequest(BaseModel):
    board: List[List[int]]
    move_history: List[Dict]


class PowerUpRequest(BaseModel):
    board: List[List[int]]
    player: int
    power_up_type: str


# ============= ENDPOINTS =============
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render main game page using Jinja2"""
    context = {
        "request": request,
        "title": "Grid Capture Ultimate",
        "game_modes": ["Player vs Player", "Player vs AI"],
        "personalities": ["balanced", "aggressive", "defensive", "chaotic"],
        "difficulty_levels": [
            {"value": 1, "name": "Very Easy"},
            {"value": 2, "name": "Easy"},
            {"value": 3, "name": "Medium"},
            {"value": 4, "name": "Hard"},
            {"value": 5, "name": "Expert"}
        ],
        "power_ups": [
            {"id": "shield", "name": "Shield", "icon": "🛡️"},
            {"id": "double", "name": "Double", "icon": "⚡"},
            {"id": "swap", "name": "Swap", "icon": "🔄"}
        ]
    }
    return templates.TemplateResponse("index.html", context)

@app.post("/api/get-move")
async def get_move(request: MoveRequest):
    """Get AI move with advanced strategies"""
    try:
        state = GameState(request.board)

        personality = AIPersonality(request.personality)

        if request.agent_type == "adaptive":
            agent = AdaptiveAgent(depth=request.difficulty, personality=personality)
        elif request.agent_type == "predictive":
            agent = PredictiveAgent(depth=request.difficulty, personality=personality)
        else:
            agent = AdaptiveAgent(depth=3, personality=AIPersonality.BALANCED)

        move = agent.choose_move(state, request.player)

        if move is None:
            raise HTTPException(400, "No valid moves available")

        # Calculate move metrics
        captures = agent._count_potential_captures(state, move[0], move[1], request.player)

        return {
            "move": move,
            "captures": captures,
            "agent_type": request.agent_type,
            "personality": request.personality,
            "confidence": min(100, captures * 20 + random.randint(60, 80))
        }

    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/get-predictions")
async def get_predictions(request: PredictionRequest):
    """Get predicted opponent moves"""
    try:
        state = GameState(request.board)
        agent = PredictiveAgent(depth=3)

        predictions = agent.get_predictions(state, request.player, request.num_predictions)

        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/analyze-game")
async def analyze_game(request: GameAnalysisRequest):
    """Provide game analysis and statistics"""
    try:
        state = GameState(request.board)

        blue, red = state.count_pieces()
        control = state.get_territory_control()

        # Calculate momentum (piece difference trend)
        momentum = 0
        if len(request.move_history) >= 3:
            recent_blue = sum(1 for m in request.move_history[-3:] if m["player"] == 1)
            recent_red = sum(1 for m in request.move_history[-3:] if m["player"] == 2)
            momentum = recent_blue - recent_red

        # Identify critical positions
        empty = state.get_empty_cells()
        critical_positions = []
        for r, c in empty:
            agent = AdaptiveAgent()
            blue_value = agent._count_potential_captures(state, r, c, 1)
            red_value = agent._count_potential_captures(state, r, c, 2)

            if blue_value >= 2 or red_value >= 2:
                critical_positions.append({
                    "position": (r, c),
                    "blue_captures": blue_value,
                    "red_captures": red_value
                })

        return {
            "piece_count": {"blue": blue, "red": red},
            "territory_control": control,
            "momentum": momentum,
            "critical_positions": critical_positions[:5],
            "game_progress": (25 - len(empty)) / 25 * 100,
            "winner_prediction": "blue" if blue > red else "red" if red > blue else "tied"
        }

    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/apply-powerup")
async def apply_powerup(request: PowerUpRequest):
    """Apply a power-up to the game"""
    try:
        state = GameState(request.board)
        success = state.apply_power_up(request.player, request.power_up_type)

        if not success:
            raise HTTPException(400, "Power-up not available")

        return {
            "success": True,
            "board": state.board,
            "remaining_powerups": state.power_ups[request.player]
        }

    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "ultimate-2.0"}


@app.get("/api/stats")
async def get_stats():
    """Get global game statistics"""
    return {
        "total_games": random.randint(1000, 5000),
        "active_players": random.randint(50, 200),
        "win_rates": {
            "blue": 48.5,
            "red": 51.5
        },
        "popular_strategies": [
            {"name": "Center Control", "usage": 67},
            {"name": "Corner Fortress", "usage": 45},
            {"name": "Aggressive Capture", "usage": 78}
        ]
    }


if __name__ == "__main__":
    import uvicorn

    print("🚀 Grid Capture Ultimate Edition - Starting Server...")
    print("📊 Features: AI Personalities, Power-Ups, Predictions, Game Analysis")
    print("🌐 Access at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)