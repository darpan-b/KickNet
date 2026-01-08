import pygame

class Renderer:
    def __init__(self, grid_height, grid_width):
        pygame.init()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.cell_size = 50
        self.screen_width = self.grid_width * self.cell_size
        self.screen_height = self.grid_height * self.cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Soccer Game")

    def draw(self, positions, ball_position, ball_owner):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.screen.fill((0, 128, 0))  # Green field

        for r in range(self.grid_height):
            for c in range(self.grid_width):
                pygame.draw.rect(self.screen, (0, 100, 0), (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size), 1)

        for agent, (row, col) in positions.items():
            color = (255, 0, 0) if "red" in agent else (0, 0, 255)
            pygame.draw.circle(self.screen, color, (col * self.cell_size + self.cell_size // 2, row * self.cell_size + self.cell_size // 2), self.cell_size // 3)

        ball_row, ball_col = ball_position
        pygame.draw.circle(self.screen, (255, 255, 255), (ball_col * self.cell_size + self.cell_size // 2, ball_row * self.cell_size + self.cell_size // 2), self.cell_size // 4)

        pygame.display.flip()
