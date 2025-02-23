import random
import tkinter as tk
import time
import numpy as np
import tkinter as tk

# 迷路サイズ（ここを変えると迷路とキャンバスのサイズが連動）
row, col = 10, 10  
cell_size = 20  # 1セルのピクセルサイズ

# 迷路データ（1 = 壁, 0 = 通路）
maze = [[1] * (col * 2 + 1) for _ in range(row * 2 + 1)]
visited = [[False] * col for _ in range(row)]

# 方向（右, 下, 左, 上）
directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

# 初期セルをスタックに追加
stack = [((0, 0), random.sample(directions, len(directions)))]

# GUI 設定（迷路サイズに応じてキャンバスのサイズを決定）
canvas_width = (col * 2 + 1) * cell_size
canvas_height = (row * 2 + 1) * cell_size

root = tk.Tk()
root.title("迷路生成")
root.geometry(f"{canvas_width + 20}x{canvas_height + 40}")  # 余白を考慮
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

class QLearning:
    def __init__(self, maze):
        self.maze = maze
        self.row, self.col = self.maze.get_size()
        self.num_state = self.row * self.col
        self.num_action = 4
        self.Q_table = np.zeros((self.num_state, self.num_action))
        self.state = self.get_state()
        
    #ε-Greedy method
    def epsilon_greedy(self):
        if random.random()<0.1:
            return random.randint(0,3)
        else:
            return self.select_best_action()
            
        
    def select_best_action(self):
        return self.Q_table[self.state, :].argmax()

    def reward(self):
        return 0 if self.maze.is_goal() else -1

    def get_state(self):
        _, col = self.maze.get_size()
        x, y = self.maze.get_position()
        return x * col + y  # 修正: 迷路のセル数に合わせる


    def restart(self):
        self.maze.reset()
        self.state = self.get_state()

    def step(self, learning_rate, discount_rate):
        action = self.epsilon_greedy()
        flag = self.maze.move(action)
        next_state = self.get_state()
        next_action = self.select_best_action()
        if flag:
            self.Q_table[self.state, action] -=0.1
        else:
            self.Q_table[self.state, action] += learning_rate * (
            self.reward()
            + discount_rate * self.Q_table[next_state, next_action]
            - self.Q_table[self.state, action]
        )
        self.state = next_state

            
class Maze:
    def __init__(self, row=10, col = 10,canvas=None, root=None):
        self.maze = np.array(maze)
        self.goal = np.array([row*2-1, col*2-1])
        self.initPos = np.array([1, 1])
        self.position = self.initPos.copy()
        self.row = row
        self.col = col
        self.canvas = canvas
        self.root = root
        
        self.maze = self.create_maze()
        
    def get_position(self):
        return self.position

    def get_size(self):
        return row*2+1, col*2+1

    def is_goal(self):
        return np.array_equal(self.position, self.goal)

    def reset(self):
        self.position = self.initPos.copy()
        
    def set_pos(self,x,y):
        self.position = np.array([x,y])
        

    def move(self, action):
        
        actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
        dx, dy = actions[action]  # 選択されたアクションに対応する移動量
        x = self.position[0] + dx
        y = self.position[1] + dy

        # 壁にぶつからないかチェック
        if self.maze[x, y] == 0:
            self.set_pos(x,y)
            return False
        return True
        
        
    def create_maze(self):
        maze = np.ones((self.row * 2 + 1, self.col * 2 + 1))
        visited = np.zeros((self.row, self.col), dtype=bool)
        queue = [tuple([self.initPos[0]-1, self.initPos[1]-1 ])]
        
        while queue:
            x, y = queue.pop(0)
            visited[y][x] = True
            maze[y * 2 + 1, x * 2 + 1] = 0
            
            directions = random.sample([(0, 1), (1, 0), (0, -1), (-1, 0)],4)
            time.sleep(0.05)
            self.draw_maze(maze)
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.col and 0 <= ny < self.row and not visited[ny][nx]:
                    maze[y * 2 + 1 + dy, x * 2 + 1 + dx] = 0
                    visited[ny][nx] = True
                    queue.append((nx, ny))
                    self.draw_maze(maze)
        
        i = 0
        while i <40:
            s,t = (random.randint(1,19), random.randint(1,19))
            if maze[s,t] == 1:
                maze[s,t] = 0
                i+=1
            
                
        return maze
    
    def draw_maze(self, maze):
        
        """ 現在の迷路状態を描画する """
        canvas.delete("all")  # 画面をクリア
        
            
        for i in range(len(maze)):
            for j in range(len(maze[i])):
                color = "white" if maze[i][j] == 0 else "black"
                canvas.create_rectangle(
                    j * cell_size, i * cell_size,
                    j * cell_size + cell_size, i * cell_size + cell_size,
                    fill=color, outline="black"
                )
        # TkinterのGUIを更新
        self.root.update_idletasks()
        self.root.update()
    
    def draw_maze2(self, maze):
 
        """ 現在の迷路状態を描画する """
        canvas.delete("all")  # 画面をクリア
        x = self.get_position()[0]
        y = self.get_position()[1]
        
            
        for i in range(len(maze)):
            for j in range(len(maze[i])):
                if x==i and y ==j:
                    color ="red"
                else: color = "white" if maze[i][j] == 0 else "black"
                canvas.create_rectangle(
                    j * cell_size, i * cell_size,
                    j * cell_size + cell_size, i * cell_size + cell_size,
                    fill=color, outline="black"
                )
        # TkinterのGUIを更新
        self.root.update_idletasks()
        self.root.update()

epocs = 1000
steps = 6000
learning_rate = 0.1
discount_rate = 0.95
maze = Maze(row, col, canvas, root)
q_learn = QLearning(maze)

for epoc in range (epocs):
    step =0
    q_learn.restart()
    while not maze.is_goal() and step<steps:
        q_learn.step(learning_rate,discount_rate)
        maze.draw_maze2(maze.maze)
        step+=1

        print("\x1b[K")
        print(f"episode : {epoc} step : {step} ")
        print(f"position: {maze.position}")
        

# Tkinter メインループ
root.mainloop()
