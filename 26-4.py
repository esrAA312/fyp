import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Parameters
GAMMA = 0.99
LR = 0.0005
MIN_BATCH_SIZE = 32
MEMORY_SIZE = 10000
EPSILON = 0.1
EPSILON_MAX = 1.0
EPSILON_GROWTH = 1.001
EPSILON_MIN = 0.0
EPSILON_DECAY = 0.999
TARGET_UPDATE = 1
NUM_EPISODES = 10
NUM_DEVICES = 4
FEATURES_PER_DEVICE = 3
INPUT_DIM = (NUM_DEVICES, FEATURES_PER_DEVICE)
DESIRED_ACCURACY = 0.8
MAX_ITERATIONS = 5
ALPHA_N = 3.0
ALPHA_E = 2.0
ALPHA_L = 2.0
L_MAX = 500
G = 7000
TAU = 1e-28
MU_MB = 1
MU_BITS = MU_MB * 8 * 1e6
D = 20 * 1e6
LAMBDA = 1
ENERGY_SCALE = 1e12

# CNN for DDQN
# 64 X 64 X 64 
class CNN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)  # من الطبقة الطيّ إلى 64 وحدة
        self.fc2 = nn.Linear(64, 64)          # 64 إلى 64 وحدة
        self.fc3 = nn.Linear(64, 64)          # 64 إلى 64 وحدة
        self.fc4 = nn.Linear(64, output_dim)  # من 64 إلى فضاء الإجراء

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # تسطيح الإدخال
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # بدون تنشيط في طبقة الإخراج
        return x

# Global Model for federated learning
class GlobalModel(nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool2(x)
        # tells PyTorch to automatically calculate the size of the second dimension so that the
        # total number of elements in the tensor remains the same.
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# FiveGNetwork
class FiveGNetwork:
    def __init__(self, base_bandwidth=100, latency=0.001, capacity=10):
        self.base_bandwidth = base_bandwidth
        self.latency = latency
        self.capacity = capacity
        self.connected_devices = 0

    def connect_device(self):
        if self.connected_devices < self.capacity:
            self.connected_devices += 1
            variation = np.random.uniform(0.9, 1.1)
            effective_bandwidth = self.base_bandwidth * (1 - 0.05 * self.connected_devices) * variation
            return effective_bandwidth, self.latency
        else:
            print("5G Network: Capacity exceeded, using fallback bandwidth.")
            return 10, 0.01

    def disconnect_device(self):
        if self.connected_devices > 0:
            self.connected_devices -= 1

# EdgeDevice
class EdgeDevice:
    def __init__(self, id, cpu_freq, energy, bandwidth, fiveg_network):
        self.id = id
        self.cpu_freq = cpu_freq
        self.energy = energy
        self.base_bandwidth = bandwidth
        self.fiveg_network = fiveg_network
        self.model = GlobalModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # دالة خسارة تُستخدم عادةً في مهام التصنيف متعدد الفئات
        self.criterion = nn.CrossEntropyLoss()
        self.local_data = None
        self.effective_bandwidth, self.latency = self.fiveg_network.connect_device()


    def set_local_data(self, data):
        if self.local_data is None:
            self.local_data = data
            print(f"Device {self.id}: Assigned {len(data) if data else 0} samples")
        else:
            print(f"Device {self.id}: Data already assigned, skipping")

    def charge_energy(self, w=1):
        return np.random.poisson(w)

    def train_local_model(self, epochs, energy_rate=1):
        if self.local_data is None:
            print(f"Device {self.id}: No data assigned, skipping training.")
            return self.model.state_dict(), 0, 0, 0

        device = torch.device("cpu")
        self.model.to(device)
        self.model.train()
        loader = torch.utils.data.DataLoader(self.local_data, batch_size=85, shuffle=True, drop_last=False)

        for epoch in range(epochs):
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

        # Equation 4 from the paper
        T_local = MU_BITS * G / self.cpu_freq if self.cpu_freq > 0 else float('inf')
        # Equation 10 from the paper
        # Energy scale is used
        # يُحسب B_k لتقييم استهلاك الطاقة أثناء تدريب النموذج المحلي على الجهاز، ويُستخدم لتحديد ما إذا كانت الطاقة المتوفرة كافية للتدريب ولتحديث حالة الطاقة
        # لتحويل وحدة الطاقة إلى نطاق مناسب وتجنب مشاكل الدقة العددية، مما يجعل قيمة B_k أسهل في التعامل والتفسير
        B_k = (self.cpu_freq ** 2) * TAU * MU_BITS * G * ENERGY_SCALE
        # Poisson distribution for energy charging
        C_k = self.charge_energy(energy_rate)
        print(f"Device {self.id}: Charging with {C_k} energy units.")
        if self.energy < B_k:
            print(f"Device {self.id}: Insufficient energy ({self.energy:.2f} < {B_k:.2f}), skipping training.")
            return self.model.state_dict(), 0, 0, 0
        # Equation 11 from the paper
        self.energy = max(self.energy - B_k + C_k, 0)
        # Equation 5 from the paper (D = 20 * 1e6)
        T_trans = D / self.effective_bandwidth if self.effective_bandwidth > 0 else float('inf')
        return self.model.state_dict(), T_local, T_trans, B_k

    def report_resources(self):
        return {'cpu_freq': self.cpu_freq, 'energy': self.energy, 'bandwidth': self.effective_bandwidth}

    def __del__(self):
        self.fiveg_network.disconnect_device()

# MECServer
class MECServer:
    def __init__(self, num_devices=NUM_DEVICES):
        self.fiveg_network = FiveGNetwork()
        self.global_model = GlobalModel()
        self.devices = []
        for i in range(num_devices):
            cpu_freq = np.random.uniform(0, 1)
            energy = np.random.uniform(2, 5)
            bandwidth = np.random.uniform(0, 2)
            device = EdgeDevice(i, cpu_freq, energy, bandwidth, self.fiveg_network)
            self.devices.append(device)
        self.energy_history = []
        self.bandwidth_history = []
        self.data_distributed = False
        print(f"Created {len(self.devices)} unique devices")

    def distribute_data(self, trainset):
        if self.data_distributed:
            print("Data already distributed, skipping")
            return
        print(f"Starting data distribution for {len(trainset)} samples")
        data_size = len(trainset)
        data_per_device = data_size // len(self.devices)
        #يُستخدم لتوزيع العينات الزائدة بشكل عادل على الأجهزة الأولى
        remainder = data_size % len(self.devices)
        indices = list(range(data_size))
        random.shuffle(indices)
        start_idx = 0
        for i, device in enumerate(self.devices):
            # Calculate the end index for the current device عشان ما تتكرر البيانات 
            end_idx = start_idx + data_per_device + (1 if i < remainder else 0)
            # Assign a subset of data to the device 
            device_indices = indices[start_idx:end_idx]
            subset = torch.utils.data.Subset(trainset, device_indices)
            device.set_local_data(subset)
            start_idx = end_idx
        print(f"Distributed {data_size} samples across {len(self.devices)} devices")
        self.data_distributed = True

    def fed_avg(self, local_weights):

        avg_weights = {key: torch.zeros_like(local_weights[0][key]) for key in local_weights[0]}
        num_clients = len(local_weights)
        for weights in local_weights:
            for key in weights:
                avg_weights[key] += weights[key] / num_clients
        return avg_weights

    def simulate_training_round(self, selected_devices, epochs=3):
        local_weights = []
        total_energy = 0
        total_bandwidth = 0
        delays = []
        print(f"Selected devices: {selected_devices}")

        for device_id in selected_devices:
            device = self.devices[device_id]
            total_bandwidth += device.effective_bandwidth  # Always record bandwidth
            weights, T_local, T_trans, energy_used = device.train_local_model(epochs)
            print(f"Device {device_id}: Energy used: {energy_used} pJ, Bandwidth: {device.effective_bandwidth:.2f} Mbps")
            delay = T_local + T_trans
            delays.append(delay)
            if energy_used > 0:
                local_weights.append(weights)
                total_energy += energy_used

        if local_weights:
            new_weights = self.fed_avg(local_weights)
            # Update global model with new weights
            # تقوم بتحميل الأوزان أو المعاملات المخزنة في قاموس إلى النموذج العصبي لتحديث حالته
            self.global_model.load_state_dict(new_weights)
        self.energy_history.append(total_energy)
        self.bandwidth_history.append(total_bandwidth)
        print(f"Round: Total Energy={total_energy:.2f}, Total Bandwidth={total_bandwidth:.2f}")
        max_latency = max(delays) if delays else 0
        # تقوم بحساب الحد الأقصى للتأخير (max_latency) من قائمة التأخيرات (delays) التي تم جمعها أثناء التدريب المحلي
        return max_latency, total_energy, total_bandwidth

    def evaluate_global_model(self, testloader):
        self.global_model.eval()
        correct = 0
        total = 0
        device = torch.device("cpu")
        #  move the model (self.global_model) to a specified device, such as a CPU or GPU
        # تقوم بتقييم نموذج عالمي باستخدام . تضع النموذج في وضع التقييم، تنقله إلى جهاز محدد (مثل CPU)، وتحسب عدد التنبؤات الصحيحة والإجمالية
        self.global_model.to(device)
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                outputs = self.global_model(data)
                # 1: يشير إلى البعد (dimension) الذي نريد أخذ القيمة القصوى منه.
                # _: يتم تجاهل القيم القصوى نفسها، ونأخذ فقط الفئات المتوقعة (predicted).
                _, predicted = torch.max(outputs.data, 1)
                # retrieves the size of the first dimension (dimension 0) of the target tensor,
                # which typically represents the batch size (i.e., the number of samples in the current batch)
                # useful for calculating metrics like accuracy, where you need the total number of samples.
                total += target.size(0)
                # تقوم بحساب عدد التنبؤات الصحيحة من خلال مقارنة القيم المتوقعة (predicted) مع القيم الحقيقية (target).
                # تقوم بجمع عدد التنبؤات الصحيحة في المتغير correct
                correct += (predicted == target).sum().item()
        accuracy = correct / total
        return accuracy

# DeviceSelectionEnv
class DeviceSelectionEnv:
    def __init__(self, mec_server):
        self.mec_server = mec_server
        self.num_devices = len(mec_server.devices)
        self.e_max_per_device = np.array([device.energy for device in mec_server.devices])        
        self.e_max_total = np.sum(self.e_max_per_device)
        self.action_space = [i for i in range(self.num_devices)]

    def generate_state(self):
        state = np.array([list(device.report_resources().values()) for device in self.mec_server.devices])
        return state

    def step(self, action):
        selected_indices = [i for i, val in enumerate(action) if val == 1]
        num_selected = len(selected_indices)

        max_latency, total_energy_consumed, total_bandwidth = self.mec_server.simulate_training_round(selected_indices)
##!!!!!!!!
        self.e_max_total = np.sum([device.energy for device in self.mec_server.devices])

        # Having a reward in negative means the negative sides of the equation is bigger, 
        # Later on we will choose the biggest reward to select the devices
        reward = (ALPHA_N * (num_selected / NUM_DEVICES)
                  - ALPHA_E * (total_energy_consumed / self.e_max_total if self.e_max_total > 0 else 1.0)
                  - ALPHA_L * (max_latency / L_MAX))

        state = self.generate_state()
        print(f"Step: Reward={reward:.2f}, Energy={total_energy_consumed:.2f}, Bandwidth={total_bandwidth:.2f}")
        return state, reward

    def reset(self):
        return self.generate_state()

# ReplayMemory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, min_batch_size):
        return random.sample(self.memory, min_batch_size)

    def __len__(self):
        return len(self.memory)

# DDQNAgent
class DDQNAgent:
    def __init__(self, input_dim, output_dim):
        self.policy_net = CNN(1, output_dim)
        self.target_net = CNN(1, output_dim)
        # If we don't like the loss or the reward we can change the optimizer to lr to 0.001 or something else
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.epsilon = EPSILON

    def preprocess_state(self, state):
        state_array = np.array(state).flatten()
        target_size = 256 # 16 × 16
        # This Diemnsion can be changed when we change the input size of the CNN if we got the error
        # وتحولها إلى مصفوفة مسطحة، ثم تملأها بالأصفار لتصل إلى حجم 256
        padded_state = np.pad(state_array, (0, target_size - len(state_array)), mode='constant', constant_values=0)
        return padded_state.reshape(1, 1, 16, 16)

# If a random number (between 0 and 1) is less than self.epsilon, the agent explores an action
    def select_action(self, state):
        n = len(state)  # عدد الأجهزة الكلي (n)
        action = [0] * n  # تهيئة فضاء الإجراء بـ 0 لكل عميل
        if random.random() < self.epsilon:
            # استكشاف: اختيار عدد عشوائي من العملاء (بين 1 و n)
            num_selected = random.randint(1, n)
            selected_indices = random.sample(range(n), num_selected)
            for idx in selected_indices:
                action[idx] = 1
        else:
            # استغلال: اختيار العملاء بناءً على الشبكة
            with torch.no_grad():
                state_tensor = torch.tensor(self.preprocess_state(state), dtype=torch.float32)
                scores = self.policy_net(state_tensor).squeeze()
                with open("scores_output.txt", "a") as f:
                    f.write(f"Scores: {scores.tolist()}\n")

                # اختيار أعلى num_selected عملاء (يمكن أن يكون أقل من n)
                num_selected = random.randint(1, n)  # أو قيمة ثابتة مثل n//2
                selected_indices = torch.argsort(scores, descending=True)[:num_selected].tolist()
                for idx in selected_indices:
                    action[idx] = 1
        return action

    def optimize_model(self):
        if len(self.memory) < MIN_BATCH_SIZE:
            return
        batch = self.memory.sample(MIN_BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(np.stack([self.preprocess_state(s) for s in states]), dtype=torch.float32)
        next_states = torch.tensor(np.stack([self.preprocess_state(s) for s in next_states]), dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        q_values = self.policy_net(states)
        next_q_values_policy = self.policy_net(next_states).detach()
        next_q_values_target = self.target_net(next_states).detach()
        expected_q_values = []
        predicted_q_values = []
        for i in range(MIN_BATCH_SIZE):
            action_indices = [idx for idx, val in enumerate(actions[i]) if val == 1]
            q_pred = q_values[i][action_indices]
            #حساب القيم المستهدفة في الكود يعتمد على معادلة بيلمان المعدلة لـ DDQN
            q_next = next_q_values_target[i][torch.argmax(next_q_values_policy[i])]
            predicted_q_values.append(q_pred.mean())
            expected_q_values.append(rewards[i] + GAMMA * q_next)
        predicted_q_values = torch.stack(predicted_q_values)
        expected_q_values = torch.stack(expected_q_values)
        loss = nn.MSELoss()(predicted_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Main Simulation
if __name__ == "__main__":
    print("Starting program...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=85, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=85, shuffle=False, num_workers=2)

    mec_server = MECServer()
    mec_server.distribute_data(trainset)
    env = DeviceSelectionEnv(mec_server)
    agent = DDQNAgent(INPUT_DIM, NUM_DEVICES)
    rewards_per_episode = []
    episode_energy_history = []
    episode_bandwidth_history = []

    for episode in range(NUM_EPISODES):
        print(f"\nStarting Episode {episode+1}")
        state = env.reset()
        total_reward = 0
        iteration = 0
        avg_reward = float('inf')
        accuracy = 0.0
        episode_energy = 0
        episode_bandwidth = 0
        episode_iterations = 0

        while accuracy < DESIRED_ACCURACY and iteration < MAX_ITERATIONS:
            action = agent.select_action(state)
            next_state, reward = env.step(action)
            agent.memory.push((state, action, reward, next_state))
            agent.optimize_model()
            state = next_state
            total_reward += reward
            iteration += 1
            episode_iterations += 1
            # -1 is used to retrieve the most recent (last) energy value from the energy_history list of the mec_server object.
            # This is likely because the energy_history list stores a sequence of energy values, and the last value represents the most up-to-date or relevant data.
            episode_energy += mec_server.energy_history[-1] if mec_server.energy_history else 0
            episode_bandwidth += mec_server.bandwidth_history[-1] if mec_server.bandwidth_history else 0
            avg_reward = total_reward / iteration if iteration > 0 else reward

            accuracy = mec_server.evaluate_global_model(testloader)
            print(f"Iteration {iteration}, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, Accuracy: {accuracy:.4f}")

        agent.update_epsilon()
        if episode % TARGET_UPDATE == 0:
            # Update the target network every TARGET_UPDATE episodes
            agent.update_target_network()

        rewards_per_episode.append(total_reward)
        episode_energy_history.append(episode_energy / episode_iterations if episode_iterations > 0 else 0)
        episode_bandwidth_history.append(episode_bandwidth / episode_iterations if episode_iterations > 0 else 0)
        print(f"Episode {episode+1} Finished, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, Iterations: {iteration}, Accuracy: {accuracy:.2f}, Energy: {episode_energy_history[-1]:.2f}, Bandwidth: {episode_bandwidth_history[-1]:.2f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(rewards_per_episode, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Performance")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(episode_energy_history, label="Energy Consumption")
    plt.xlabel("Episode")
    plt.ylabel("Energy (pJ)")
    plt.title("Energy Consumption Over Episodes")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(episode_bandwidth_history, label="Bandwidth Usage")
    plt.xlabel("Episode")
    plt.ylabel("Bandwidth (Mbps)")
    plt.title("Bandwidth Usage Over Episodes")
    plt.legend()

    plt.tight_layout()
    plt.show()
