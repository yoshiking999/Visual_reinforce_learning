import random
import tkinter as tk
import time
import numpy as np
from PIL import Image, ImageTk

row, col = 10, 10

cell_size = 2

maze = [[1] * (col * 2 + 1) for _ in range(row * 2 + 1)]
visited = [[False] * col for _ in range(row)]

directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

stack = [((0, 0), random.sample(directions, len(directions)))]

canvas_width = (col * 2 + 1) * cell_size
canvas_height = (row * 2 + 1) * cell_size

root = tk.Tk()
root.title("迷路生成")
root.geometry(f"{canvas_width + 20}x{canvas_height + 40}")
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

def draw_maze():
    canvas.delete("all")
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            color = "white" if maze[i][j] == 0 else "black"
            canvas.create_rectangle(
            j * cell_size, i * cell_size,
            j * cell_size + cell_size, i * cell_size + cell_size,
            fill=color, outline="black"
            )
            
            
class QLearning:
    def __init__(self, maze_obj):

        self.maze = maze_obj
        self.row, self.col = self.maze.get_size()
        self.num_state = self.row * self.col
        self.num_action = 4
        self.Q_table = np.zeros((self.num_state, self.num_action))
        self.state = self.get_state()

    def select_best_action(self):
        """
        Q テーブルから最も値の高い行動を返す。
        """
        return int(self.Q_table[self.state, :].argmax())

    def reward(self):
        """
        報酬関数：ゴールなら 0，そうでなければ -1。
        """
        return 0 if self.maze.is_goal() else -1

    def get_state(self):
        """
        論理座標 (x, y) を状態番号として返す。
        state = x * col + y
        """
        _, c = self.maze.get_size()
        x, y = self.maze.get_position()
        return x * c + y

    def restart(self):
        """
        エピソードの最初で迷路をリセットし、状態を更新する。
        """
        self.maze.reset()
        self.state = self.get_state()

    def step(self, learning_rate, discount_rate):
        """
        1ステップ実行し、Q テーブルを更新する。
        SARSA に基づき next_action も同じポリシーで選択する。
        """
        action = self.select_best_action()
        self.maze.move(action)
        next_state = self.get_state()
        next_action = self.select_best_action()

        # 報酬と次状態の Q 値を用いて更新
        self.Q_table[self.state, action] += learning_rate * (
            self.reward()
            + discount_rate * self.Q_table[next_state, next_action]
            - self.Q_table[self.state, action]
        )
        # 状態を更新
        self.state = next_state
    
class Maze:
    def init(self, maze_array):
        self.maze = np.array(maze_array)
        # ゴール座標（※論理座標(9,9)を仮定）
        self.goal = np.array([9, 9])
        # エージェントの初期位置（論理座標）
        self.initPos = np.array([0, 0])
        self.position = self.initPos.copy()
            # ロボット画像の読み込み（パスは環境に合わせて変更）
        self.original_image = Image.open(r"C:\Users\yoshi\OneDrive\デスクトップ\RL_Maize\robo.png")
        # 画像サイズを適当にリサイズ（ここでは 200x200）
        self.resized_image = self.original_image.resize((200, 200))
        self.img = ImageTk.PhotoImage(self.resized_image)

    def get_position(self):
        """
        エージェントの論理座標 (x, y) を返す。
        """
        return self.position

    def get_size(self):
        """
        迷路の論理サイズ(row, col)を返す。
        """
        return row, col

    def is_goal(self):
        """
        現在位置がゴールかどうかを返す。
        """
        return np.array_equal(self.position, self.goal)

    def reset(self):
        """
        エージェントの位置を初期化。
        """
        self.position = self.initPos.copy()

    def move(self, action):
        """
        指定された行動（0=上、1=右、2=下、3=左）に基づきエージェントを1マス移動する。
        下層迷路配列で 1（壁）なら移動不可、0（通路）なら移動可。
        """
        actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dx, dy = actions[action]

        # 移動先（論理座標）
        x_next = self.position[0] + dx
        y_next = self.position[1] + dy

        # 壁チェック (maze[x, y] == 1 なら行けない)
        if self.maze[x_next, y_next] == 1:
            # 移動不可
            return self.position
        else:
            # 移動可能
            self.position = np.array([x_next, y_next])
            return self.position

    def draw_maze(self):
        """
        現在の迷路とエージェント位置を描画する。
        """
        canvas.delete("all")
        # 迷路セルを描画
        for i in range(len(self.maze)):
            for j in range(len(self.maze[i])):
                color = "white" if self.maze[i][j] == 0 else "black"
                canvas.create_rectangle(
                    j * cell_size, i * cell_size,
                    j * cell_size + cell_size, i * cell_size + cell_size,
                    fill=color, outline="black"
                )
        # エージェント（ロボット画像）を配置
        x_screen = self.position[1] * cell_size + cell_size // 2
        y_screen = self.position[0] * cell_size + cell_size // 2
        canvas.create_image(x_screen, y_screen, image=self.img, anchor=tk.CENTER)
        
def create_maze():
    """
    再帰的バックトラッキング法で迷路を生成。
    完了後、 2*(row)+1 x 2*(col)+1 の maze を返す。
    """
    while stack:
        (x, y), dir_list = stack[-1]
        visited[y][x] = True
        # 論理座標 (x,y) に対応する maze 上のマスを通路に
        maze[y * 2 + 1][x * 2 + 1] = 0
        # 描画更新
        draw_maze()
        root.update()
        time.sleep(0.02)
        while dir_list:
            dx, dy = dir_list.pop()
            nx, ny = x + dx, y + dy
            if 0 <= nx < col and 0 <= ny < row and not visited[ny][nx]:
                # 壁を削って通路に
                maze[y * 2 + 1 + dy][x * 2 + 1 + dx] = 0
                stack.append(((nx, ny), random.sample(directions, len(directions))))
                break
            else:
                stack.pop()
    return maze
    
epocs = 1000 # 総エピソード数
steps = 1000 # 1エピソードでの最大ステップ数
learning_rate = 0.1
discount_rate = 0.95
sleep_time = 0.5

generated_maze = create_maze()
    
# Maze クラスを生成（論理移動で使用）
maze_obj = Maze(generated_maze)

# Q 学習クラスのインスタンスを生成
q_learn = QLearning(maze_obj)

# エピソードループ
for epoc in range(epocs):
    step_count = 0
    # 毎エピソードで迷路をリセット
    q_learn.restart()

    # ゴール到達 or 指定ステップ数まで
    while (not maze_obj.is_goal()) and (step_count < steps):
        q_learn.step(learning_rate, discount_rate)
        # 迷路を描画
        maze_obj.draw_maze()
        step_count += 1
        # 処理を少し待機
        time.sleep(sleep_time)

    # 学習の進捗をコンソール出力
    print("\x1b[K", end="")  # 画面上書き用のエスケープシーケンス
    print(f"episode : {epoc} step : {step_count} ")

# ウィンドウを閉じるまで待機
root.mainloop()