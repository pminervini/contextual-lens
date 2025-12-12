#!/usr/bin/env python3
# tetris.py
# Minimal Tetris clone using pygame
# Author: Mercury (Inception)

import pygame
import random
import sys

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
WIDTH, HEIGHT = 300, 600          # Window size
BLOCK_SIZE = 30                   # Size of a single tetromino block
COLS, ROWS = 10, 20               # Grid dimensions
FPS = 10                           # Frames per second (controls speed)

# Colors (RGB)
BLACK   = (0, 0, 0)
WHITE   = (255, 255, 255)
GRAY    = (128, 128, 128)
COLORS  = [
    (0, 255, 255),   # I
    (0, 0, 255),     # J
    (255, 165, 0),   # L
    (255, 255, 0),   # O
    (0, 255, 0),     # S
    (128, 0, 128),   # T
    (255, 0, 0)      # Z
]

# ----------------------------------------------------------------------
# Tetromino definitions
# ----------------------------------------------------------------------
# Each piece is a list of 4x4 matrices (one per rotation)
TETROMINOS = {
    'I': [
        [[0,0,0,0],
         [1,1,1,1],
         [0,0,0,0],
         [0,0,0,0]],
        [[0,0,1,0],
         [0,0,1,0],
         [0,0,1,0],
         [0,0,1,0]],
    ],
    'J': [
        [[1,0,0],
         [1,1,1],
         [0,0,0]],
        [[0,1,1],
         [0,1,0],
         [0,1,0]],
        [[0,0,0],
         [1,1,1],
         [0,0,1]],
        [[0,1,0],
         [0,1,0],
         [1,1,0]],
    ],
    'L': [
        [[0,0,1],
         [1,1,1],
         [0,0,0]],
        [[0,1,0],
         [0,1,0],
         [0,1,1]],
        [[0,0,0],
         [1,1,1],
         [1,0,0]],
        [[1,1,0],
         [0,1,0],
         [0,1,0]],
    ],
    'O': [
        [[1,1],
         [1,1]],
    ],
    'S': [
        [[0,1,1],
         [1,1,0],
         [0,0,0]],
        [[0,1,0],
         [0,1,1],
         [0,0,1]],
    ],
    'T': [
        [[0,1,0],
         [1,1,1],
         [0,0,0]],
        [[0,1,0],
         [0,1,1],
         [0,1,0]],
        [[0,0,0],
         [1,1,1],
         [0,1,0]],
        [[0,1,0],
         [1,1,0],
         [0,1,0]],
    ],
    'Z': [
        [[1,1,0],
         [0,1,1],
         [0,0,0]],
        [[0,0,1],
         [0,1,1],
         [0,1,0]],
    ],
}

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def rotate(shape):
    """Rotate a 2D shape clockwise."""
    return [list(row) for row in zip(*shape[::-1])]

def valid_position(grid, shape, offset):
    """Check if shape can be placed at offset without collision."""
    off_x, off_y = offset
    for y, row in enumerate(shape):
        for x, cell in enumerate(row):
            if cell:
                new_x, new_y = off_x + x, off_y + y
                if new_x < 0 or new_x >= COLS or new_y >= ROWS:
                    return False
                if new_y >= 0 and grid[new_y][new_x]:
                    return False
    return True

def clear_lines(grid):
    """Clear completed lines and return number of lines cleared."""
    new_grid = [row for row in grid if any(cell == 0 for cell in row)]
    lines_cleared = ROWS - len(new_grid)
    # Add empty rows on top
    for _ in range(lines_cleared):
        new_grid.insert(0, [0]*COLS)
    return new_grid, lines_cleared

# ----------------------------------------------------------------------
# Game classes
# ----------------------------------------------------------------------
class Piece:
    """Current falling tetromino."""
    def __init__(self, shape_key):
        self.shape_key = shape_key
        self.rotations = TETROMINOS[shape_key]
        self.rotation = 0
        self.shape = self.rotations[self.rotation]
        self.color = COLORS[list(TETROMINOS.keys()).index(shape_key)]
        self.x = COLS // 2 - len(self.shape[0]) // 2
        self.y = -2  # start above the visible grid

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.rotations)
        self.shape = self.rotations[self.rotation]

    def rotate_back(self):
        self.rotation = (self.rotation - 1) % len(self.rotations)
        self.shape = self.rotations[self.rotation]

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def get_cells(self):
        """Return list of (x, y) positions occupied by the piece."""
        cells = []
        for y, row in enumerate(self.shape):
            for x, cell in enumerate(row):
                if cell:
                    cells.append((self.x + x, self.y + y))
        return cells

class Tetris:
    """Main game logic."""
    def __init__(self):
        self.grid = [[0]*COLS for _ in range(ROWS)]
        self.current_piece = self.new_piece()
        self.next_piece = self.new_piece()
        self.score = 0
        self.game_over = False

    def new_piece(self):
        shape_key = random.choice(list(TETROMINOS.keys()))
        return Piece(shape_key)

    def lock_piece(self):
        """Freeze current piece into the grid."""
        for x, y in self.current_piece.get_cells():
            if y >= 0:
                self.grid[y][x] = self.current_piece.color
        self.grid, lines = clear_lines(self.grid)
        self.score += lines * 100
        self.current_piece = self.next_piece
        self.next_piece = self.new_piece()
        if not valid_position(self.grid, self.current_piece.shape, (self.current_piece.x, self.current_piece.y)):
            self.game_over = True

    def move_piece(self, dx, dy):
        self.current_piece.move(dx, dy)
        if not valid_position(self.grid, self.current_piece.shape, (self.current_piece.x, self.current_piece.y)):
            self.current_piece.move(-dx, -dy)
            return False
        return True

    def rotate_piece(self):
        self.current_piece.rotate()
        if not valid_position(self.grid, self.current_piece.shape, (self.current_piece.x, self.current_piece.y)):
            self.current_piece.rotate_back()
            return False
        return True

    def hard_drop(self):
        while self.move_piece(0, 1):
            pass
        self.lock_piece()

# ----------------------------------------------------------------------
# Rendering functions
# ----------------------------------------------------------------------
def draw_grid(surface, grid):
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell:
                pygame.draw.rect(surface, cell,
                                 (x*BLOCK_SIZE, y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(surface, GRAY,
                                 (x*BLOCK_SIZE, y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)

def draw_piece(surface, piece):
    for x, y in piece.get_cells():
        if y >= 0:
            pygame.draw.rect(surface, piece.color,
                             (x*BLOCK_SIZE, y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(surface, GRAY,
                             (x*BLOCK_SIZE, y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)

def draw_next_piece(surface, piece):
    font = pygame.font.SysFont('Arial', 18)
    label = font.render('Next:', True, WHITE)
    surface.blit(label, (WIDTH + 10, 10))
    for x, y in piece.get_cells():
        px = WIDTH + 10 + (x - piece.x) * BLOCK_SIZE
        py = 30 + (y - piece.y) * BLOCK_SIZE
        pygame.draw.rect(surface, piece.color,
                         (px, py, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(surface, GRAY,
                         (px, py, BLOCK_SIZE, BLOCK_SIZE), 1)

def draw_score(surface, score):
    font = pygame.font.SysFont('Arial', 18)
    label = font.render(f'Score: {score}', True, WHITE)
    surface.blit(label, (WIDTH + 10, 200))

# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH + 150, HEIGHT))
    pygame.display.set_caption('Tetris')
    clock = pygame.time.Clock()

    game = Tetris()
    fall_time = 0
    fall_speed = 0.5  # seconds per fall

    while True:
        dt = clock.tick(FPS) / 1000
        fall_time += dt

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    game.move_piece(-1, 0)
                if event.key == pygame.K_RIGHT:
                    game.move_piece(1, 0)
                if event.key == pygame.K_DOWN:
                    game.move_piece(0, 1)
                if event.key == pygame.K_UP:
                    game.rotate_piece()
                if event.key == pygame.K_SPACE:
                    game.hard_drop()

        # Automatic piece fall
        if fall_time > fall_speed:
            fall_time = 0
            if not game.move_piece(0, 1):
                game.lock_piece()

        # Drawing
        screen.fill(BLACK)
        draw_grid(screen, game.grid)
        draw_piece(screen, game.current_piece)
        draw_next_piece(screen, game.next_piece)
        draw_score(screen, game.score)

        if game.game_over:
            font = pygame.font.SysFont('Arial', 36)
            label = font.render('Game Over', True, WHITE)
            screen.blit(label, (WIDTH//2 - label.get_width()//2, HEIGHT//2 - label.get_height()//2))

        pygame.display.update()

if __name__ == '__main__':
    main()
