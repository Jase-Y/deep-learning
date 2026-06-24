import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import signal
import sys
import asyncio
import json
import time
import random
import websockets
from pathlib import Path
from collections import defaultdict

# ============================================================
# 教学配置：通过修改这个变量，自由切换不同的循环神经网络进行对比
# 可选值: 'RNN' (标准循环神经网络), 'GRU' (门控循环单元), 'LSTM' (长短期记忆网络)
# ============================================================
MODEL_TYPE = 'LSTM'  # 可修改为 'RNN' 或 'GRU'

# 文件路径配置 - 支持单文件或多文件
DATA_FOLDER = r"E:\TimeGAN-master\TimeGAN-master\data\stock_data.csv"  # 直接使用当前 CSV 文件作为数据来源

# 动态生成文件名，防止不同模型的权重互相覆盖
CHECKPOINT_PATH = f'sleep_{MODEL_TYPE.lower()}_checkpoint.pth'
FINAL_WEIGHTS_PATH = f'sleep_{MODEL_TYPE.lower()}_weights.pth'

# WebSocket服务器配置
HOST = "127.0.0.1"
PORT = 8765

# 训练参数
SEQUENCE_LENGTH = 20  # 时间步长
BATCH_SIZE = 32
LEARNING_RATE = 0.001  # 降低学习率以适应真实数据
NUM_EPOCHS = 200
TRAIN_SPLIT = 0.8  # 训练集比例


# ============================================================
# 1. 信号捕获与安全性配置
# ============================================================
def receive_signal(signum, frame):
    print(f"\n[警告] 收到资源回收信号 (Signal: {signum})! 正在紧急保存当前进度...")
    global model, optimizer, epoch, loss
    save_checkpoint(epoch, model, optimizer, loss, path=CHECKPOINT_PATH)
    print(f"[退出] {MODEL_TYPE} 模型的进度已安全保存，程序优雅退出。")
    sys.exit(0)


signal.signal(signal.SIGTERM, receive_signal)
signal.signal(signal.SIGINT, receive_signal)


def save_checkpoint(epoch, model, optimizer, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"-> 检查点已保存至: {path}")


# ============================================================
# 2. 核心教学知识点：多网络统一实现
# ============================================================
class SleepRNNDemo(nn.Module):
    def __init__(self, cell_type='LSTM', input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(SleepRNNDemo, self).__init__()
        self.cell_type = cell_type.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 教学重点：对比三种核心时序组件的API和内部结构
        if self.cell_type == 'RNN':
            self.rnn_core = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif self.cell_type == 'GRU':
            self.rnn_core = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif self.cell_type == 'LSTM':
            self.rnn_core = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("未知的网络类型！请选择 'RNN', 'GRU' 或 'LSTM'")

        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)  # 添加dropout防止过拟合

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        if self.cell_type == 'LSTM':
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            out, (hn, cn) = self.rnn_core(x, (h0, c0))
        else:
            out, hn = self.rnn_core(x, h0)

        # 只取最后一个时间步的输出
        last_output = out[:, -1, :]
        last_output = self.dropout(last_output)  # 应用dropout
        return self.fc(last_output)


# ============================================================
# 3. 多文件数据预处理模块
# ============================================================
def resolve_csv_files(data_source):
    """将单个 CSV 文件或目录解析为 CSV 文件列表"""
    path = Path(data_source)
    if path.is_file() and path.suffix.lower() == '.csv':
        return [path]

    if path.is_dir():
        csv_files = list(path.glob('*.csv'))
        csv_files.extend(list(path.glob('*.CSV')))
        return sorted(set(csv_files))

    raise ValueError(f"无法解析数据源: {data_source}")


class MultiFileTimeSeriesProcessor:
    def __init__(self, folder_path, sequence_length=20, train_split=0.8):
        self.folder_path = folder_path
        self.sequence_length = sequence_length
        self.train_split = train_split
        self.file_data = {}  # 存储每个文件的数据
        self.all_prices = []  # 所有价格数据（用于整体训练）
        self.file_names = []  # 文件名称列表

    def scan_csv_files(self):
        """扫描数据源中的所有CSV文件"""
        csv_files = resolve_csv_files(self.folder_path)

        if not csv_files:
            raise ValueError(f"在 {self.folder_path} 中未找到任何CSV文件")

        print(f"找到 {len(csv_files)} 个CSV文件:")
        for f in csv_files:
            print(f"  - {f.name}")

        return csv_files

    def load_file_data(self, file_path):
        """加载单个CSV文件并提取价格数据"""
        try:
            df = pd.read_csv(file_path)

            # 尝试找到价格列
            price_cols = ['Close', 'close', 'CLOSE', 'Adj_Close', 'Adj Close', 'AdjClose', 'adj_close', 'Price', 'price']
            price_data = None

            for col in price_cols:
                if col in df.columns:
                    price_data = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=np.float32)
                    break

            if price_data is None:
                # 尝试使用第一个数值列
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    price_data = pd.to_numeric(df[numeric_cols[0]], errors='coerce').to_numpy(dtype=np.float32)
                    print(f"  自动选择数值列: {numeric_cols[0]}")
                else:
                    print(f"  警告: 文件 {file_path.name} 没有找到价格列，跳过")
                    return None

            # 尝试按日期排序
            date_cols = ['DATE', 'Date', 'date', 'Datetime', 'datetime', 'Timestamp', 'timestamp']
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df = df.sort_values(col)
                    # 重新获取排序后的价格
                    for pcol in price_cols:
                        if pcol in df.columns:
                            price_data = pd.to_numeric(df[pcol], errors='coerce').to_numpy(dtype=np.float32)
                            break
                    break

            # 清理数据
            price_data = price_data[~np.isnan(price_data)]  # 移除NaN
            price_data = price_data[~np.isinf(price_data)]  # 移除无穷值

            if len(price_data) < self.sequence_length + 1:
                print(f"  警告: 文件 {file_path.name} 数据量不足 ({len(price_data)} 条)，跳过")
                return None

            return price_data

        except Exception as e:
            print(f"  错误: 加载文件 {file_path.name} 失败: {e}")
            return None

    def load_all_data(self):
        """加载所有CSV文件的数据"""
        csv_files = self.scan_csv_files()

        for file_path in csv_files:
            print(f"\n加载文件: {file_path.name}")
            price_data = self.load_file_data(file_path)

            if price_data is not None:
                self.file_data[file_path.stem] = price_data
                self.all_prices.extend(price_data)
                self.file_names.append(file_path.stem)
                print(f"  成功加载 {len(price_data)} 个数据点")

        if not self.file_data:
            raise ValueError("没有成功加载任何CSV文件的数据")

        # 将所有价格合并为一个数组（整体训练）
        self.all_prices = np.array(self.all_prices, dtype=np.float32)
        print(f"\n总数据点: {len(self.all_prices)}")
        print(f"价格范围: {self.all_prices.min():.6f} - {self.all_prices.max():.6f}")

        return self.file_data

    def create_sequences(self, prices):
        """从价格数据创建序列"""
        X, y = [], []
        for i in range(self.sequence_length, len(prices)):
            X.append(prices[i - self.sequence_length:i])
            y.append(prices[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def prepare_training_data(self):
        """准备训练数据（使用所有数据）"""
        X, y = self.create_sequences(self.all_prices)

        # 分割训练集和测试集
        train_size = int(len(X) * self.train_split)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        print(f"\n训练数据形状: X={X_train.shape}, y={y_train.shape}")
        print(f"测试数据形状: X={X_test.shape}, y={y_test.shape}")

        return X_train, X_test, y_train, y_test

    def get_file_statistics(self):
        """获取每个文件的统计信息"""
        stats = {}
        for name, data in self.file_data.items():
            stats[name] = {
                'count': len(data),
                'min': data.min(),
                'max': data.max(),
                'mean': data.mean(),
                'std': data.std()
            }
        return stats


# ============================================================
# 4. 训练函数（支持多文件）
# ============================================================
def train_model():
    """执行模型训练（使用所有CSV文件数据）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"======== 教学实验: 当前正在训练 [{MODEL_TYPE}] 模型 ========")
    print(f"运行设备: {device}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 加载多文件数据
    print("\n[1] 扫描并加载所有CSV文件...")
    processor = MultiFileTimeSeriesProcessor(
        folder_path=DATA_FOLDER,
        sequence_length=SEQUENCE_LENGTH,
        train_split=TRAIN_SPLIT
    )

    file_data = processor.load_all_data()
    stats = processor.get_file_statistics()

    print("\n[2] 文件统计信息:")
    for name, stat in stats.items():
        print(f"  {name}: 数据量={stat['count']}, 均值={stat['mean']:.6f}, 标准差={stat['std']:.6f}")

    # 准备训练数据
    X_train, X_test, y_train, y_test = processor.prepare_training_data()

    # 转换为PyTorch张量
    X_train_t = torch.FloatTensor(X_train).unsqueeze(-1).to(device)  # [batch, seq_len, 1]
    y_train_t = torch.FloatTensor(y_train).unsqueeze(-1).to(device)  # [batch, 1]
    X_test_t = torch.FloatTensor(X_test).unsqueeze(-1).to(device)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(-1).to(device)

    print(f"\n[3] 训练数据准备完成")

    # 实例化模型
    global model, optimizer, loss, epoch
    model = SleepRNNDemo(
        cell_type=MODEL_TYPE,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=1
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    start_epoch = 0
    total_epochs = NUM_EPOCHS
    loss = torch.tensor(0.0)
    best_test_loss = float('inf')
    best_train_loss = float('inf')

    # 断点续训
    if os.path.exists(CHECKPOINT_PATH):
        print(f"发现 [{MODEL_TYPE}] 的历史训练记录，正在恢复...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print(f"成功恢复！从第 {start_epoch} 个 Epoch 继续。")

    print(f"\n[4] 开始训练 ({total_epochs} 轮)...")

    try:
        for epoch in range(start_epoch, total_epochs):
            model.train()

            # 批量训练
            total_loss = 0.0
            num_batches = 0

            # 打乱训练数据
            indices = torch.randperm(len(X_train_t))

            for i in range(0, len(X_train_t), BATCH_SIZE):
                batch_indices = indices[i:i + BATCH_SIZE]
                batch_X = X_train_t[batch_indices]
                batch_y = y_train_t[batch_indices]

                outputs = model(batch_X)

                # 确保形状匹配
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(1)
                if batch_y.dim() == 1:
                    batch_y = batch_y.unsqueeze(1)

                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_train_loss = total_loss / num_batches if num_batches > 0 else 0

            # 测试
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_t)
                test_loss = criterion(test_outputs, y_test_t).item()

            # 学习率调整
            scheduler.step(test_loss)

            # 保存最佳模型
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), f'{MODEL_TYPE.lower()}_best.pth')

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss

            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'[{MODEL_TYPE}] Epoch [{epoch + 1}/{total_epochs}], '
                      f'Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss:.6f}, '
                      f'LR: {current_lr:.6f}')
                save_checkpoint(epoch, model, optimizer, loss, path=CHECKPOINT_PATH)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f"\n[成功] {MODEL_TYPE} 模型训练完成！")
        print(f"最佳训练损失: {best_train_loss:.6f}")
        print(f"最佳测试损失: {best_test_loss:.6f}")

        # 保存最终权重
        torch.save(model.state_dict(), FINAL_WEIGHTS_PATH)
        print(f"部署权重已保存至: {FINAL_WEIGHTS_PATH}")

        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)

    except Exception as e:
        print(f"训练发生意外: {e}")
        save_checkpoint(epoch, model, optimizer, loss, path=CHECKPOINT_PATH)
        raise e

    return model, processor


# ============================================================
# 5. WebSocket 遥测数据流服务（多文件轮询）
# ============================================================
async def stream_data(websocket):
    """WebSocket数据流服务 - 多文件轮询"""
    print(f"\n[终端接入] 客户端已连接。开始下发 {MODEL_TYPE} 遥测数据...")

    # 初始化模型并加载权重
    model = SleepRNNDemo(
        cell_type=MODEL_TYPE,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=1
    )

    try:
        # 尝试加载最佳模型，如果不存在则加载最终模型
        best_path = f'{MODEL_TYPE.lower()}_best.pth'
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=torch.device('cpu')))
            print(f"[系统] 成功加载最佳权重文件: {best_path}")
        elif os.path.exists(FINAL_WEIGHTS_PATH):
            model.load_state_dict(torch.load(FINAL_WEIGHTS_PATH, map_location=torch.device('cpu')))
            print(f"[系统] 成功加载权重文件: {FINAL_WEIGHTS_PATH}")
        else:
            print(f"[警告] 未找到权重文件，将使用未经训练的初始权重。")
    except Exception as e:
        print(f"[警告] 加载权重失败: {e}")

    model.eval()

    # 加载所有CSV文件数据用于流式预测
    all_prices = []
    file_names = []

    try:
        csv_files = resolve_csv_files(DATA_FOLDER)

        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)

                # 提取价格列
                price_cols = ['Close', 'close', 'CLOSE', 'Adj_Close', 'Adj Close', 'AdjClose', 'adj_close', 'Price', 'price']
                for col in price_cols:
                    if col in df.columns:
                        prices = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=np.float32)
                        prices = prices[~np.isnan(prices)]
                        if len(prices) > SEQUENCE_LENGTH:
                            all_prices.extend(prices.tolist())
                            file_names.append(file_path.stem)
                            print(f"[系统] 加载文件 {file_path.name}: {len(prices)} 个数据点")
                        break
            except Exception as e:
                print(f"[警告] 读取文件 {file_path.name} 失败: {e}")

        if not all_prices:
            raise ValueError("没有加载到任何有效数据")

        print(f"[系统] 总共加载了 {len(all_prices)} 个数据点")

    except Exception as e:
        print(f"[警告] 无法读取数据文件: {e}")
        # 生成模拟数据
        np.random.seed(42)
        t = np.linspace(0, 100, 3000)
        all_prices = 0.7 + 0.01 * np.sin(t * 0.1) + 0.005 * np.random.randn(len(t))
        all_prices = np.clip(all_prices, 0.65, 0.75)
        all_prices = all_prices.tolist()

    # 添加当前文件索引，用于轮询显示
    current_file_idx = 0

    try:
        with torch.no_grad():
            # 使用滑动窗口进行预测
            for i in range(SEQUENCE_LENGTH, len(all_prices) - 1):
                start_time = time.time()

                # 获取序列窗口
                window = all_prices[i - SEQUENCE_LENGTH:i]
                input_tensor = torch.FloatTensor(window).view(1, -1, 1)

                # 预测
                pred = model(input_tensor).item()
                actual = all_prices[i]

                # 计算延迟
                calc_time = (time.time() - start_time) * 1000
                simulated_latency = calc_time + random.uniform(5.0, 15.0)

                # 轮询显示当前数据来源
                if i % 100 == 0:
                    current_file_idx = (current_file_idx + 1) % len(file_names) if file_names else 0
                    current_file = file_names[current_file_idx] if file_names else "unknown"

                # 打包数据
                payload = {
                    "timestamp": time.time() * 1000,
                    "model_type": MODEL_TYPE,
                    "ch1_actual": float(actual),
                    "ch2_predict": float(pred),
                    "error_abs": abs(float(actual) - float(pred)),
                    "latency_ms": round(simulated_latency, 2),
                    "data_source": file_names[current_file_idx] if file_names else "synthetic",
                    "data_index": i
                }

                await websocket.send(json.dumps(payload))
                await asyncio.sleep(0.03)  # 控制采样率

    except Exception as e:
        print(f"[断开] 客户端连接中断或发生异常: {e}")


async def start_server():
    """启动WebSocket服务器"""
    async with websockets.serve(stream_data, HOST, PORT):
        print("===============================================")
        print(f"  [SYS] 工业级边缘计算遥测终端已启动")
        print(f"  [SYS] 当前挂载计算核心: {MODEL_TYPE} 神经网络")
        print(f"  [SYS] 数据源: {DATA_FOLDER} (所有CSV文件)")
        print(f"  [SYS] 监听端口: ws://{HOST}:{PORT}")
        print("===============================================")
        print("等待前端监控面板接入...")
        await asyncio.Future()


# ============================================================
# 6. 主程序
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='多文件AUDCAD价格预测系统')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['train', 'serve', 'both'],
                        help='运行模式: train(仅训练), serve(仅服务), both(训练+服务)')
    parser.add_argument('--model', type=str, default='LSTM',
                        choices=['RNN', 'GRU', 'LSTM'],
                        help='选择模型类型')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--folder', type=str, default=None,
                        help='CSV文件路径或文件夹路径 (覆盖默认)')

    args = parser.parse_args()

    # 更新配置
    MODEL_TYPE = args.model
    NUM_EPOCHS = args.epochs
    if args.folder:
        DATA_FOLDER = args.folder

    CHECKPOINT_PATH = f'sleep_{MODEL_TYPE.lower()}_checkpoint.pth'
    FINAL_WEIGHTS_PATH = f'sleep_{MODEL_TYPE.lower()}_weights.pth'

    if args.mode in ['train', 'both']:
        # 执行训练
        model, processor = train_model()

        if args.mode == 'train':
            print("\n训练完成，程序退出。")
            sys.exit(0)

    if args.mode in ['serve', 'both']:
        # 启动WebSocket服务器
        print("\n启动遥测数据服务...")
        asyncio.run(start_server())
