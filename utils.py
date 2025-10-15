import os
import random
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pygame
import requests


class BaseAgent(ABC):
    def __init__(self): ...

    @abstractmethod
    def policy(self, state): ...


class RandomAgent(BaseAgent):
    def policy(self, state):
        return random.choice(state.valid_actions)


class HumanAgent(BaseAgent):
    def policy(self, state):
        return int(input(f"Enter a valid action ({state.valid_actions}): "))


class OnlineAgent(BaseAgent):
    def __init__(self, level):
        self.url = "https://ludolab.net/solve/connect4"
        self.level = level
        assert level in range(1, 10 + 1)  # higher means stronger

    def policy(self, state):
        """
        The request should return a JSON like:
        ```
        [
            {'move': '4', 'score': -1.0},
            {'move': '2', 'score': -2.0},
            {'move': '3', 'score': -2.0},
            {'move': '1', 'score': -4.0},
            {'move': '5', 'score': -2.0},
            {'move': '6', 'score': -2.0},
            {'move': '7', 'score': -4.0}
        ]
        ```
        and finally return an integer action.
        """
        import random
        import time

        time.sleep(random.uniform(0.1, 0.5))

        resp = requests.get(
            self.url,
            params={
                "position": self._state_to_position(state),
                "level": self.level,
            },
            headers={
                "User-Agent": "MCTSResearchBot/1.0 (contact: shunchizhang.cs@gmail.com)",
                "Referer": "https://ludolab.net/play/four-in-a-line/computer",
                "Accept": "*/*",
            },
        )
        action = int(resp.json()[0]["move"]) - 1
        return action

    @staticmethod
    def _state_to_position(state):
        """
        return the action sequence
        """
        actions = []
        for step in range(1, state.record.max() + 1):
            row, col = np.where(state.record == step)
            assert len(row) == len(col) == 1
            actions.append(col[0] + 1)
        return "".join(map(str, actions))


def get_image(path):
    cwd = os.path.dirname(__file__)
    image = pygame.image.load(os.path.join(cwd, path))
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc


class State:
    """
    Connect 4 Game State
    """

    def __init__(self, record=None):
        if record is None:
            record = np.zeros((6, 7))
        self.record = np.array(record, dtype=int)

        self.board = np.zeros_like(self.record)
        self.board[self.record % 2 == 1] = 1
        self.board[self.record % 2 == 0] = 2
        self.board[self.record == 0] = 0

        self.current_steps = self.record.max().item()
        self.opponent_piece = (self.current_steps % 2) + 1
        self.piece = 3 - self.opponent_piece

        self.valid_actions = (self.board[0] == 0).nonzero()[0].tolist()
        self.is_terminal, self.result = self._check_termination()

    def transition(self, action):
        new_record = self.record.copy()
        # find the lowest empty row to place the piece
        for r in range(self.record.shape[0] - 1, -1, -1):
            if new_record[r, action] == 0:
                new_record[r, action] = self.current_steps + 1
                break
        return State(new_record)

    def _check_termination(self):
        num_rows, num_cols = self.board.shape

        # terminated: wins
        result = {self.piece: 1, self.opponent_piece: -1}
        # horizontal
        for r in range(num_rows):
            for c in range(num_cols - 3):
                if all(self.board[r, c + i] == self.piece for i in range(4)):
                    return True, result
        # vertical
        for r in range(num_rows - 3):
            for c in range(num_cols):
                if all(self.board[r + i, c] == self.piece for i in range(4)):
                    return True, result
        # positively sloped diagonals
        for r in range(num_rows - 3):
            for c in range(num_cols - 3):
                if all(self.board[r + i, c + i] == self.piece for i in range(4)):
                    return True, result
        # negatively sloped diagonals
        for r in range(3, num_rows):
            for c in range(num_cols - 3):
                if all(self.board[r - i, c + i] == self.piece for i in range(4)):
                    return True, result

        # terminated: draw
        if np.all(self.board != 0):
            return True, {self.piece: 0, self.opponent_piece: 0}

        # unterminated
        return False, None

    def render(self):
        # 1. get constants
        empty_tile = get_image("img/connect4/empty.png")
        red_tile = get_image("img/connect4/red.png")
        black_tile = get_image("img/connect4/black.png")

        board_padding = 4
        tile_size = empty_tile.get_size()[0]
        num_rows, num_cols = self.board.shape

        board_width = tile_size * num_cols + board_padding * 2
        board_height = tile_size * num_rows + board_padding * 2

        # 2. create simple screen
        screen = pygame.Surface((board_width, board_height))
        screen.fill("#0C2A89")
        screen_inside = pygame.Surface(
            (
                board_width - 2 * (board_padding - 1),
                board_height - 2 * (board_padding - 1),
            )
        )
        screen_inside.fill("#1D3DAD")
        screen.blit(screen_inside, (board_padding - 1, board_padding - 1))

        # 3. blit tiles
        for i in range(0, num_rows):
            for j in range(0, num_cols):
                if self.board[i, j] == 0:
                    tile = empty_tile
                elif self.board[i, j] == 1:
                    tile = red_tile
                elif self.board[i, j] == 2:
                    tile = black_tile
                screen.blit(
                    tile,
                    (
                        j * tile_size + board_padding,
                        i * tile_size + board_padding,
                    ),
                )

        # 4. scale the screen
        font_size = screen_scaling = 9
        screen = pygame.transform.scale(
            screen, (board_width * screen_scaling, board_height * screen_scaling)
        )

        # 5. overlay record numbers
        pygame.font.init()
        font = pygame.font.Font(None, font_size * screen_scaling)
        for i in range(0, num_rows):
            for j in range(0, num_cols):
                if self.record[i, j] != 0:
                    text = font.render(str(self.record[i, j]), True, (255, 255, 255))
                    text_rect = text.get_rect(
                        center=(
                            (j * tile_size + board_padding + tile_size // 2)
                            * screen_scaling,
                            (i * tile_size + board_padding + tile_size // 2)
                            * screen_scaling,
                        )
                    )
                    screen.blit(text, text_rect)

        observation = np.array(pygame.surfarray.pixels3d(screen))
        observation = np.transpose(observation, axes=(1, 0, 2))

        plt.imshow(observation)
        plt.axis("off")
        plt.show()
