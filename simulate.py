import tkinter as tk
import threading
import flwr as fl


class MyFlowerServer(fl.server.Server):
    def __init__(self):
        super().__init__(
            server_address=("127.0.0.1:8080"),
            num_rounds=10,
            timeout=60,
            grace_period=5,
            min_fit_clients=1,
            max_fit_clients=10,
            min_available_clients=1,
            sample_fraction=1.0,
            desired_accuracy=0.99,
            report=("loss", "accuracy", "precision", "recall"),
            metrics_func=None,
            secure=False,
            config={"bandwidth": 1000000, "resources": {"cpu": 1, "gpu": 0}},
        )

    def get_weights(self):
        return self.aggregator.aggregate()

    def fit(self, weights):
        return self.aggregator.evaluate(weights)

    def evaluate(self, weights):
        return self.aggregator.evaluate(weights)


class MyFlowerClient(fl.client.NumPyClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = {"bandwidth": 100000, "resources": {"cpu": 1, "gpu": 0}}

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.config = config
        return self.model.evaluate(parameters, config=self.config)

    def evaluate(self, parameters, config):
        self.config = config
        return self.model.evaluate(parameters, config=self.config)


def start_server():
    server = MyFlowerServer()
    server.start()


def start_client(**kwargs):
    fl.client.start_numpy_client("0.0.0.0:8080", client=client)
    # fl.client.start_numpy_client()


class ClientConfigGUI:
    def __init__(self, root, client):
        self.root = root
        self.client = client
        self.root.title("Client Configuration")

        self.bandwidth_label = tk.Label(root, text="Bandwidth (bps):")
        self.bandwidth_label.grid(row=0, column=0, padx=5, pady=5)
        self.bandwidth_entry = tk.Entry(root)
        self.bandwidth_entry.grid(row=0, column=1, padx=5, pady=5)

        self.cpu_label = tk.Label(root, text="CPU Cores:")
        self.cpu_label.grid(row=1, column=0, padx=5, pady=5)
        self.cpu_entry = tk.Entry(root)
        self.cpu_entry.grid(row=1, column=1, padx=5, pady=5)

        self.gpu_label = tk.Label(root, text="GPU Cores:")
        self.gpu_label.grid(row=2, column=0, padx=5, pady=5)
        self.gpu_entry = tk.Entry(root)
        self.gpu_entry.grid(row=2, column=1, padx=5, pady=5)

        self.save_button = tk.Button(root, text="Save", command=self.save_config)
        self.save_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

    def save_config(self):
        bandwidth = int(self.bandwidth_entry.get())
        cpu_cores = int(self.cpu_entry.get())
        gpu_cores = int(self.gpu_entry.get())

        self.client.config["bandwidth"] = bandwidth
        self.client.config["resources"]["cpu"] = cpu_cores
        self.client.config["resources"]["gpu"] = gpu_cores


if __name__ == "__main__":
    # 启动服务端
    server_thread = threading.Thread(target=start_server)
    server_thread.start()

    # 创建客户端实例
    client = MyFlowerClient()

    # 启动 GUI
    root = tk.Tk()
    app = ClientConfigGUI(root, client)

    # 启动客户端
    client_thread = threading.Thread(target=start_client, kwargs={"client": client})
    client_thread.start()

    root.mainloop()
