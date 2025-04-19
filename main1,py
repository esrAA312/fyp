import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

# تحديث المعاملات بناءً على الورقة العلمية
GAMMA = 0.99  # عامل التخفيض
LR = 0.0005  # معدل التعلم
MIN_BATCH_SIZE = 32  # حجم الدفعة التدريبية
MEMORY_SIZE = 10000  # حجم الذاكرة
EPSILON = 0.1  # معدل الاستكشاف الابتدائي (ε-greedy: 0.1 → 0)
EPSILON_MAX = 1.0  # الحد الأقصى لـ Epsilon
EPSILON_GROWTH = 1.001  # معدل الزيادة التدريجي لـ Epsilon
EPSILON_MIN = 0.0  # Minimum exploration rate
EPSILON_DECAY = 0.999  # Decay factor (adjustable)
TARGET_UPDATE = 10  # عدد الحلقات قبل تحديث الشبكة الهدف
NUM_EPISODES = 3 # عدد جولات التدريب  
NUM_DEVICES = 4  # عدد الأجهزة 
NUM_SELECTED_DEVICES = 4  # عدد الأجهزة المختارة (η = 4)
FEATURES_PER_DEVICE = 3  # الميزات لكل جهاز (Latency, Bandwidth, Energy Consumption)
INPUT_DIM = (NUM_DEVICES, FEATURES_PER_DEVICE)
DESIRED_ACCURACY = 0.8  # دقة افتراضية للتوقف (يمكن تعديلها)

ALPHA_N = 3.0  # معامل عدد الأجهزة المختارة (α_n = 3)
ALPHA_E = 2.0  # معامل استهلاك الطاقة (α_e = 2)
ALPHA_L = 2.0  # معامل التأخير (α_l = 2)
E_MAX = 100  # الحد الأقصى لاستهلاك الطاقة (E_max = 100)
L_MAX = 500  # الحد الأقصى للتأخير (L_max = 500 s)
MAX_ITERATIONS = 5 # الحد الأقصى لعدد التكرارات في كل حلقة

# نموذج CNN لاختيار الأجهزة (DDQN) وللنموذج العالمي
class CNN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(CNN, self).__init__()
        # طبقة تشعبية واحدة بـ 16 فلتر بدلاً من طبقتين بـ 32
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        # تجميع أقصى لتقليل الأبعاد
        self.pool = nn.MaxPool2d(2, 2)
        # طبقة تشعبية ثانية بـ 32 فلتر بدلاً من طبقتين بـ 64
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # بعد التجميع: 32 × 8 × 8
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # طبقة وسيطة أصغر
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # 16×16×16
        x = self.pool(x)               # 8×8×16
        x = torch.relu(self.conv2(x))  # 8×8×32
        x = x.view(x.size(0), -1)      # تسطيح إلى [batch_size, 32×8×8]
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# نموذج عالمي بسيط لمحاكاة التعلم الاتحادي (Global Model) this will be edited when we have the dataset
class GlobalModel(nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        # الطبقة التشعبية الأولى: 3 قنوات إدخال (RGB)، 32 فلتر بحجم 3×3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # الطبقة التشعبية الثانية: 32 قناة إدخال، 32 فلتر بحجم 3×3
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # تجميع أقصى بحجم 2×2
        self.pool1 = nn.MaxPool2d(2, 2)
        # الطبقة التشعبية الثالثة: 32 قناة إدخال، 64 فلتر بحجم 3×3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # الطبقة التشعبية الرابعة: 64 قناة إدخال، 64 فلتر بحجم 3×3
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # تجميع أقصى بحجم 2×2
        self.pool2 = nn.MaxPool2d(2, 2)
        # طبقة كاملة الاتصال: بعد التجميع، الأبعاد تصبح 64 × 8 × 8
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 فئات لـ CIFAR-10

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # 32×32×32
        x = torch.relu(self.conv2(x))  # 32×32×32
        x = self.pool1(x)              # 16×16×32
        x = torch.relu(self.conv3(x))  # 16×16×64
        x = torch.relu(self.conv4(x))  # 16×16×64
        x = self.pool2(x)              # 8×8×64
        x = x.view(x.size(0), -1)      # تسطيح إلى [batch_size, 64×8×8]
        x = torch.relu(self.fc1(x))    # إلى 128
        x = self.fc2(x)                # إلى 10
        return x

# بيئة اختيار الأجهزة
class DeviceSelectionEnv:
    def __init__(self, num_devices=NUM_DEVICES):
        self.num_devices = num_devices
        self.state = self.generate_state()
        self.action_space = [i for i in range(num_devices)]

    def generate_state(self):
        # values from the paper we need to add and configure
        latency = np.random.rand(self.num_devices, 1) * L_MAX  # التأخير بين 0 و L_max (500 ثانية)
        bandwidth = np.random.rand(self.num_devices, 1) * 2  # عرض النطاق بين 0 و 2 ميجابت/ثانية (R = 2 Mbps)
        energy = np.random.rand(self.num_devices, 1) * E_MAX  # استهلاك الطاقة بين 0 و E_max (100)
        return np.hstack((latency, bandwidth, energy))

    def step(self, action):
        selected_indices = [i for i, val in enumerate(action) if val == 1]
        selected_devices = np.array(self.state)[selected_indices]
        num_selected = len(selected_indices)  # m: عدد الأجهزة المختارة
        # These 2 below needs to get changed into the correct formula according to the paper equations
        total_energy = np.sum(selected_devices[:, 2])  # E: مجموع استهلاك الطاقة
        max_latency = np.max(selected_devices[:, 0])  # L: أقصى تأخير

        # معادلة المكافأة بناءً على الورقة (Equation 13):
        # R(s, a) = α_n * (m/n) - α_e * (E/E_max) - α_l * (L/L_max)
        reward = (ALPHA_N * (num_selected / NUM_DEVICES)
                  - ALPHA_E * (total_energy / E_MAX)
                  - ALPHA_L * (max_latency / L_MAX))
        self.state = self.generate_state()
        done = False
        # إذا كانت المكافأة أقل من 0، فإننا نعتبرها نهاية الحلقة
        return self.state, reward, done

    def reset(self):
        self.state = self.generate_state()
        return self.state

# ذاكرة إعادة التشغيل
class ReplayMemory:
    def __init__(self, capacity):
        # Initialize the memory with a fixed capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, min_batch_size):
        return random.sample(self.memory, min_batch_size)

    def __len__(self):
        return len(self.memory)

# وكيل DDQN لاختيار العملاء
class DDQNAgent:
    def __init__(self, input_dim, output_dim):
        self.policy_net = CNN(1, output_dim)
        self.target_net = CNN(1, output_dim)
        # Adaptive Moment Estimation is an algorithm
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.epsilon = EPSILON

    def preprocess_state(self, state):
        # Convert 10×3 state to 16×16 tensor
        state_array = np.array(state)  # Shape: [N, 3] where N could be 10 or 4
        flat_state = state_array.flatten()  # Shape: [N * 3]
        target_size = 256  # 16 × 16
        padded_state = np.pad(flat_state, (0, target_size - len(flat_state)), mode='constant', constant_values=0)
        # 1 Batch size (1 sample at a time)
        # 1 Number of channels (grayscale "image")
        # 16 Height and 16 Width (16×16 image)
        return padded_state.reshape(1, 1, 16, 16)  # Shape: [1, 1, 16, 16]
        # Flatten to 30 elements and pad to 256 (16×16)
        # CNN is used to taking the input as 1×16×16

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
        else:#state.py
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

        # Convert states and next_states to 4D tensors
        states = torch.tensor(np.stack([self.preprocess_state(s) for s in states]), dtype=torch.float32)  # Shape: [32, 1, 16, 16]
        next_states = torch.tensor(np.stack([self.preprocess_state(s) for s in next_states]), dtype=torch.float32)  # Shape: [32, 1, 16, 16]
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # Predict Q-values
        q_values = self.policy_net(states)  # Shape: [MIN_BATCH_SIZE, NUM_DEVICES]
        next_q_values_policy = self.policy_net(next_states).detach()# Q(s', a'; θ)
        next_q_values_target = self.target_net(next_states).detach()# Q(s', a'; θ')

        expected_q_values = []
        predicted_q_values = []
        for i in range(MIN_BATCH_SIZE):
            action_indices = [idx for idx, val in enumerate(actions[i]) if val == 1]  # Binary action to indices
            q_pred = q_values[i][action_indices]  # Q-values of selected actions
            # Double DQN target
            #حساب القيم المستهدفة في الكود يعتمد على معادلة بيلمان المعدلة لـ DDQN
            q_next = next_q_values_target[i][torch.argmax(next_q_values_policy[i])]  # Double DQN target
            predicted_q_values.append(q_pred.mean())#ما تتوقعه الشركه العصيبه
            expected_q_values.append(rewards[i] + GAMMA * q_next)#ما يفترض ان تتوقعه بناء على بولمان

        predicted_q_values = torch.stack(predicted_q_values)
        expected_q_values = torch.stack(expected_q_values)
        # يقوم بحساب الفرق (الخسارة) بين القيم المتوقعة (predicted_q_values) والقيم المتوقعة الحقيقية 
        # (expected_q_values) باستخدام دالة الخسارة MSE (متوسط مربع الخطأ). ثم يقوم بتهيئة المُحسّن (optimizer) لتصفير التدرجات، ويحسب التدرجات العكسية للخسارة،
        # وأخيرًا يحدّث معايير النموذج بناءً على هذه التدرجات لتحسين الأداء

        loss = nn.MSELoss()(predicted_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # تقليل الاستكشاف العشوائي تدريجيًا مع تحسن أداء الوكيل (agent) وزيادة الاعتماد على السياسة المستفادة
    def update_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# تعريف الدوال
def local_training(model, trainloader, num_epochs=3):
    # تعريف معيار الخسارة (loss function) باستخدام CrossEntropyLoss من مكتبة PyTorch.
    # هذا المعيار يُستخدم عادةً في مسائل التصنيف لقياس الفرق بين التوقعات والقيم الحقيقية
    criterion = nn.CrossEntropyLoss()
    # model.parameters(): يمرر جميع المعاملات القابلة للتدريب في النموذج
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # Batch normalization a layer that allows every layer of the network to do learning more independently.
    # It is used to normalize the output of the previous layers.
    # Dropout is a regularization technique to prevent overfitting, Dropouts are the regularization technique that is used to prevent overfitting in the model.
    # It is added to randomly switching some percentage of neurons of the network
    # We use the .train() method to make sure these two layers aren't misbehaving during training.
    model.train()
    print("Starting local training...")
    for epoch in range(num_epochs):
        # الدالة enumerate هي دالة مدمجة في Python تُستخدم لإضافة عداد (counter) إلى كائن قابل للتكرار.
        # بدلاً من أن تُعيد فقط العناصر من الكائن القابل للتكرار (مثل trainloader)، فإنها تُعيد زوجًا (tuple)
        for i, data in enumerate(trainloader):
            inputs, labels = data
            # تُعاد تعيين التدرجات (gradients) إلى الصفر لتجنب تراكمها من الخطوات السابقة.
            optimizer.zero_grad()
            # السطر ده مفروض يتغير لم ندخل ليهو البيانات من الاجهزة المشاركة في التدريب
            # الخرج (outputs) هو ما يتنبأ به النموذج بناءً على أوزانه الحالية
            #  التسمية (labels) هي الحقيقة التي نريد من النموذج أن يتعلمها
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # يتم حساب التدرجات (gradients) لجميع المعاملات في النموذج باستخدام الانتشار العكسي (backpropagation)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:  # طباعة تقدم كل 10 دفعات
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")
    return model.state_dict()

# This function is gonna be edited later on
def aggregate_weights(global_model, local_weights, client_data_sizes=None):
    global_dict = global_model.state_dict()
    num_clients = len(local_weights)
    if client_data_sizes is None:
        client_data_sizes = [5000 // num_clients] * num_clients
    total_data_size = sum(client_data_sizes)
    
    for key in global_dict.keys():
        stacked_weights = torch.stack([weights[key] for weights in local_weights], dim=0)
        # Equation 1
        weighted_sum = sum(client_data_sizes[k] * stacked_weights[k] for k in range(num_clients))
        global_dict[key] = weighted_sum / total_data_size
    # يحمل القاموس المحدث إلى النموذج العام.
    global_model.load_state_dict(global_dict)
    return global_model

def evaluate_accuracy(model, testloader):
    # يضع النموذج في وضع التقييم (بدلاً من التدريب)، مما يعني أنه لن يتم تعديل أوزانه أو تحديثها
    model.eval()
    # لحساب عدد التنبؤات الصحيحة
    correct = 0
    # لحساب العدد الإجمالي للعينات
    total = 0
    # لا نريد حساب التدرجات أثناء التقييم، لذا نستخدم torch.no_grad()
    # هذا يساعد في تقليل استخدام الذاكرة وزيادة السرعة
    with torch.no_grad():
        for data in testloader:
            # الصور التي سيتم اختبار النموذج عليها
            # التسميات الحقيقية (الإجابات الصحيحة) لتلك الصور
            images, labels = data
            outputs = model(images)
            # 1: يشير إلى البعد (dimension) الذي نريد أخذ القيمة القصوى منه.
            # _: يتم تجاهل القيم القصوى نفسها، ونأخذ فقط الفئات المتوقعة (predicted).
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Evaluated Accuracy: {accuracy:.4f}")
    return accuracy

# The start of the program
if __name__ == '__main__':
    print("Starting program...")
    transform = transforms.Compose([
        # PIL inside ToTensor Enabled Python to deal with images
        # قيم تقريبية لحساب الانحراف المتوسط و المعياري
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # تحميل مجموعة بيانات CIFAR-10
    # for training which is 50,000 images
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # تحديد حزمة معينة من البيانات (5000صورة)
    # لتقليل وقت التدريب
    subset_indices = list(range(5000))
    trainset = torch.utils.data.Subset(trainset, subset_indices)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=85, shuffle=True, num_workers=2)

    # for testing which is 10,000 images
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=85, shuffle=False, num_workers=2)

    env = DeviceSelectionEnv()
    agent = DDQNAgent(INPUT_DIM, NUM_DEVICES)
    global_model = GlobalModel()
    rewards_per_episode = []

    for episode in range(NUM_EPISODES):
        print(f"\nStarting Episode {episode+1}")
        # giving the environment a new state
        # to start with in each episode
        state = env.reset()
        total_reward = 0
        iteration = 0
        avg_reward = float('inf')
        accuracy = 0.0

        while accuracy < DESIRED_ACCURACY and iteration < MAX_ITERATIONS:
            action = agent.select_action(state)
            next_state, reward, _ = env.step(action)
            agent.memory.push((state, action, reward, next_state))
            agent.optimize_model()
            state = next_state
            total_reward += reward
            iteration += 1
            avg_reward = total_reward / iteration if iteration > 0 else reward

            local_weights = []
            # Train only selected devices (where action[i] == 1)
            selected_indices = [i for i, val in enumerate(action) if val == 1]
            for client_id in selected_indices:
                local_model = GlobalModel()
                # عند استدعاء global_model.state_dict()، يتم استخراج جميع الأوزان والمعاملات القابلة للتعلم في النموذج العام كقاموس
                local_model.load_state_dict(global_model.state_dict())
                # the devices uses certain data to train the model (1MB) from the dataset only
                local_weights.append(local_training(local_model, trainloader))

            if local_weights:  # Ensure there are weights to aggregate
                print(f"Pre-Aggregation Accuracy: {evaluate_accuracy(global_model, testloader):.4f}")
                global_model = aggregate_weights(global_model, local_weights)
                print(f"Post-Aggregation Accuracy: {evaluate_accuracy(global_model, testloader):.4f}")
                accuracy = evaluate_accuracy(global_model, testloader)
            else:
                print("No devices selected, skipping aggregation.")
            print(f"Iteration {iteration}, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}")

        agent.update_epsilon()
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        rewards_per_episode.append(total_reward)
        print(f"Episode {episode+1} Finished, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, Iterations: {iteration}, Accuracy: {accuracy:.2f}, Selected Devices: {action}")

    torch.save(agent.policy_net.state_dict(), "ddqn_model.txt")
    torch.save(global_model.state_dict(), "global_model.txt")

    plt.figure(figsize=(10, 5))
    plt.plot(rewards_per_episode, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Performance")
    plt.legend()
    plt.show()
