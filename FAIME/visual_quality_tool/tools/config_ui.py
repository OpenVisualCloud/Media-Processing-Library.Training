# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import sys
import json
import threading
import webbrowser
import tkinter as tk

from pathlib import Path
from tkinter import ttk
from tkinter.filedialog import askdirectory, askopenfilename

from config_generator import generate_config


def launch_tool(data_json):
    root_dir = Path(__file__).resolve().parents[1]

    config_json = generate_config(data_json, True)
    config_str = "var cached_data = " + json.dumps(config_json, indent=4,
                                                   sort_keys=True)

    with open(str(root_dir / "config.js"), "w") as config_file:
        config_file.write(config_str)

    webbrowser.open(str("file:" / Path(root_dir / "index.html")))


def populate_tree(tree, path: Path):
    tree.delete(*tree.get_children())
    skip_dirs = ["ref-exr", "ref-png", "dist-exr", "dist-png", "diffs", "heatmaps"]
    for p1 in path.iterdir():
        if p1.is_dir() and p1.stem not in skip_dirs:
            paths = []
            for p2 in p1.iterdir():
                if p2.is_dir() and p2.stem not in skip_dirs:
                    paths.append(p2)
            if paths:
                for path in paths:
                    tree.insert("", "end", text=p1.stem + "/" + path.stem,
                                values=[path, "file"])
            else:
                tree.insert("", "end", text=p1.stem, values=[p1, "file"])


def clear_tree(tree):
    tree.delete(*tree.get_children())


class RedirectText(object):
    def __init__(self, text_ctrl):
        self.output = text_ctrl

    def write(self, string):
        self.output.insert(tk.END, string)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.seqs_tree = ttk.Treeview(self, columns="fullpath")
        self.models_tree = ttk.Treeview(self, columns="fullpath")

        self.seqs_tree.heading("#0", text="Sequences", anchor='w')
        self.models_tree.heading("#0", text="Models", anchor='w')

        self.seqs_tree.bind('<ButtonRelease-1>', self.build_models_tree)

        data_label = tk.Label(self, text="data.json")
        log_label = tk.Label(self, text="Log")

        self.data_window = tk.Text(self, width=50)
        self.log_window = tk.Text(self, width=50)

        redirect = RedirectText(self.log_window)
        sys.stdout = redirect

        btn_frame1 = tk.Frame(self)
        btn_frame2 = tk.Frame(self)

        add_data_btn = tk.Button(btn_frame1, text="Add data",
                                 command=self.add_data)
        remove_data_btn = tk.Button(btn_frame1, text="Remove data",
                                    command=self.remove_data)
        save_data_btn = tk.Button(btn_frame1, text="Save data",
                                  command=self.save_data)
        launch_tool_btn = tk.Button(btn_frame2, text="Launch tool",
                                    command=self.launch_tool)

        self.seqs_tree.grid(column=0, row=0, sticky='nswe')
        self.models_tree.grid(column=1, row=0, sticky='nswe')

        btn_frame1.grid(column=1, row=1, sticky='w')
        btn_frame2.grid(column=1, row=1, sticky='e')

        add_data_btn.grid(column=0, row=0)
        remove_data_btn.grid(column=1, row=0)
        save_data_btn.grid(column=2, row=0)
        launch_tool_btn.grid(column=0, row=0)

        data_label.grid(column=0, row=1, sticky='w')
        log_label.grid(column=1, row=1, sticky='w')

        self.data_window.grid(column=0, row=2)
        self.log_window.grid(column=1, row=2)

        menu = tk.Menu(self, tearoff=0)
        submenu = tk.Menu(menu, tearoff=0)
        submenu.add_command(label='Choose report...',
                            command=self.choose_report)
        submenu.add_command(label='Open data.json',
                            command=self.choose_data)
        submenu.add_command(label='Exit',
                            command=self.destroy)
        menu.add_cascade(label="File", menu=submenu)
        self.config(menu=menu)

        self.data = {
            "sequences": []
        }

    def launch_tool(self):
        data_json = json.loads(json.dumps(self.data))
        th = threading.Thread(target=launch_tool, args=(data_json,))
        th.start()

    def get_data(self):
        self.data_window.delete(1.0, tk.END)

        sequence = self.seqs_tree.item(
            self.seqs_tree.selection())["text"]
        model = self.models_tree.item(
            self.models_tree.selection())["text"]

        return [sequence, model]

    def add_data(self):
        seq_name, model_name = self.get_data()

        for seq in self.data["sequences"]:
            if seq["name"] == seq_name:
                if model_name not in seq['models']:
                    seq['models'].append(model_name)
                self.data_window.insert(tk.INSERT,
                                        json.dumps(self.data, indent=4))
                return

        self.data['sequences'].append({
            "name": seq_name,
            "models": [model_name]
        })
        self.data_window.insert(tk.INSERT, json.dumps(self.data, indent=4))

    def remove_data(self):
        seq_name, model_name = self.get_data()

        for idx, seq in enumerate(self.data["sequences"]):
            if seq["name"] == seq_name:
                if model_name in seq['models']:
                    seq['models'].remove(model_name)
            if not seq['models']:
                self.data["sequences"].pop(idx)

        self.data_window.insert(tk.INSERT, json.dumps(self.data, indent=4))

    def save_data(self):
        data_file = str(self.data["path"]) + "/data.json"
        self.log_window.insert(tk.INSERT, 'Data file generated: ' +
                               data_file + '\n')
        with open(Path(data_file), 'w') as outfile:
            json.dump(self.data, outfile, indent=4, sort_keys=True)

    def choose_report(self):
        self.data_window.delete(1.0, tk.END)
        self.data = {"sequences": [], "path": askdirectory()}
        self.build_seq_tree()

    def choose_data(self):
        self.data_window.delete(1.0, tk.END)

        input_data = askopenfilename()
        with open(input_data) as data_file:
            self.data = json.load(data_file)

        self.build_seq_tree()
        self.data_window.insert(tk.INSERT, json.dumps(self.data, indent=4))

    def build_seq_tree(self):
        clear_tree(self.seqs_tree)
        clear_tree(self.models_tree)
        populate_tree(self.seqs_tree, Path(str(self.data["path"])))

    def build_models_tree(self, event):
        clear_tree(self.models_tree)
        seq_dir = self.seqs_tree.item(self.seqs_tree.selection())["values"][0]
        populate_tree(self.models_tree, Path(str(seq_dir)))


if __name__ == "__main__":
    app = App()
    app.mainloop()
