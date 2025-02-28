import streamlit as st
import numpy as np
import copy
import networkx as nx
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components
import os
from jinja2 import Template

# ---------------------------
# 基础数据定义
# ---------------------------
knowledge_points = ['代数基础', '线性方程组', '二次方程', '代数表达式简化',
                      '函数的基本概念', '指数和对数', '矩阵运算', '概率论基础']
num_questions = 50  # 每个知识点50道题

dependencies = [
    ("代数基础", "线性方程组"),
    ("线性方程组", "代数表达式简化"),
    ("二次方程", "代数表达式简化"),
    ("代数表达式简化", "函数的基本概念"),
    ("函数的基本概念", "指数和对数"),
    ("指数和对数", "矩阵运算"),
    ("矩阵运算", "概率论基础")
]

prereq_map = {kp: [] for kp in knowledge_points}
for pre, post in dependencies:
    if post in prereq_map:
        prereq_map[post].append(pre)

# ---------------------------
# BKTSimulation 类定义
# ---------------------------
class BKTSimulation:
    def __init__(self, knowledge_points, num_questions, init_mastery, p_guess, p_slip, p_learn):
        self.knowledge_points = knowledge_points
        self.num_questions = num_questions
        self.init_mastery = init_mastery
        self.p_guess = p_guess      # 猜测正确概率
        self.p_slip = p_slip        # 失误概率
        self.p_learn = p_learn      # 学习率
        self.prereq_map = prereq_map
        self.reset_state()
    
    def reset_state(self):
        self.current_mastery = {kp: self.init_mastery for kp in self.knowledge_points}
        self.question_counter = {kp: 0 for kp in self.knowledge_points}
        self.correct_counts = {kp: 0 for kp in self.knowledge_points}
        self.mastery_history = {kp: [self.init_mastery] for kp in self.knowledge_points}
        self.learning_path = []
        self.last_update = None
        self.current_recommendation = None
        
        # 固定的 DAG 布局：使用 Graphviz 的 dot 布局，从上到下排列（TB）
        G = nx.DiGraph()
        for kp in self.knowledge_points:
            G.add_node(kp)
        for edge in dependencies:
            G.add_edge(edge[0], edge[1])
        try:
            self.pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB -Gnodesep=50 -Granksep=50')
        except Exception as e:
            self.pos = nx.spring_layout(G)
    
    def update_knowledge(self, p_current, correct):
        if correct:
            numerator = p_current * (1 - self.p_slip)
            denominator = p_current * (1 - self.p_slip) + (1 - p_current) * self.p_guess
        else:
            numerator = p_current * self.p_slip
            denominator = p_current * self.p_slip + (1 - p_current) * (1 - self.p_guess)
        p_post = numerator / denominator if denominator != 0 else 0
        p_new = p_post + (1 - p_post) * self.p_learn
        return p_new

    def recommend_question(self, threshold, pre_req_threshold):
        available_points = [kp for kp in self.knowledge_points if self.question_counter[kp] < self.num_questions]
        if not available_points:
            self.current_recommendation = None
            return None

        # 获取当前知识点：从当前推荐或学习路径最后一项获取
        current_kp = None
        if self.current_recommendation is not None:
            current_kp = self.current_recommendation[0]
        elif self.learning_path:
            current_kp = self.learning_path[-1]
        
        if current_kp is not None and self.current_mastery[current_kp] < threshold:
            if self.question_counter[current_kp] < self.num_questions:
                self.current_recommendation = (current_kp, self.question_counter[current_kp] + 1)
                return self.current_recommendation
        
        candidate_points = []
        if current_kp is not None:
            candidate_points = [kp for kp in available_points if current_kp in self.prereq_map.get(kp, [])]
            candidate_points = [kp for kp in candidate_points 
                                if all(self.current_mastery.get(pr, 0) >= pre_req_threshold for pr in self.prereq_map.get(kp, []))]
        if candidate_points:
            kp_selected = min(candidate_points, key=lambda kp: self.current_mastery[kp])
        else:
            eligible = []
            for kp in available_points:
                prereqs = self.prereq_map.get(kp, [])
                if prereqs:
                    if all(self.current_mastery.get(pr, 0) >= pre_req_threshold for pr in prereqs):
                        eligible.append(kp)
                else:
                    eligible.append(kp)
            if eligible:
                kp_selected = min(eligible, key=lambda kp: self.current_mastery[kp])
            else:
                kp_selected = min(available_points, key=lambda kp: self.current_mastery[kp])
        question_index = self.question_counter[kp_selected] + 1
        self.current_recommendation = (kp_selected, question_index)
        return self.current_recommendation

    def submit_answer(self, correct):
        self.last_update = {
            'current_mastery': copy.deepcopy(self.current_mastery),
            'question_counter': copy.deepcopy(self.question_counter),
            'correct_counts': copy.deepcopy(self.correct_counts),
            'mastery_history': copy.deepcopy(self.mastery_history),
            'learning_path': self.learning_path.copy(),
            'recommendation': self.current_recommendation
        }
        kp, qnum = self.current_recommendation
        old_mastery = self.current_mastery[kp]
        new_mastery = self.update_knowledge(old_mastery, correct)
        self.current_mastery[kp] = new_mastery
        self.question_counter[kp] += 1
        if correct:
            self.correct_counts[kp] += 1
        self.mastery_history[kp].append(new_mastery)
        self.learning_path.append(kp)
        self.current_recommendation = None

    def undo_last_update(self):
        if self.last_update is not None:
            self.current_mastery = self.last_update['current_mastery']
            self.question_counter = self.last_update['question_counter']
            self.correct_counts = self.last_update['correct_counts']
            self.mastery_history = self.last_update['mastery_history']
            self.learning_path = self.last_update['learning_path']
            self.current_recommendation = self.last_update['recommendation']
            self.last_update = None
            return True
        return False

    def get_pyvis_graph_html(self):
        net = Network(height="600px", width="100%", directed=True)
        net.set_options("""{
          "physics": { "enabled": false }
        }""")
        # 加载模板并转换为 Jinja2 模板对象
        template_path = os.path.join("templates", "template.html")
        if os.path.exists(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                net.template = Template(f.read())
        # 缩放 Graphviz 计算的坐标到 0-800, 0-600 区间
        xs = [coord[0] for coord in self.pos.values()]
        ys = [coord[1] for coord in self.pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        def scale_x(x):
            return 800 * (x - min_x) / (max_x - min_x) if max_x != min_x else 400
        def scale_y(y):
            return 600 * (y - min_y) / (max_y - min_y) if max_y != min_y else 300

        for node in self.knowledge_points:
            mastery = self.current_mastery[node]
            label = f"{node}\n{mastery:.2f}"
            color = px.colors.sample_colorscale("YlGnBu", mastery)[0]
            font_color = "white" if mastery >= 0.5 else "black"
            # 构造悬浮显示信息
            attempts = self.question_counter[node]
            if attempts > 0:
                accuracy = self.correct_counts[node] / attempts * 100
                accuracy_str = f"{accuracy:.1f}%"
            else:
                accuracy_str = "N/A"
            evolution = " -> ".join([f"{v:.2f}" for v in self.mastery_history[node]])
            hover_text = f"{node}\n做题数: {attempts}\n正确率: {accuracy_str}\n掌握度变化: {evolution}"
            net.add_node(node,
                         label=label,
                         title=hover_text,   # 添加悬浮显示信息
                         color=color,
                         shape="ellipse",
                         x=scale_x(self.pos[node][0]),
                         y=scale_y(self.pos[node][1]),
                         fixed=True,
                         font={"color": font_color})
        for edge in dependencies:
            net.add_edge(edge[0], edge[1])
        net.show("pyvis_graph.html")
        with open("pyvis_graph.html", "r", encoding="utf-8") as f:
            html = f.read()
        return html

    def get_learning_path_str(self):
        return " -> ".join(self.learning_path) if self.learning_path else "暂无"

# ---------------------------
# Streamlit 页面布局
# ---------------------------
st.set_page_config(page_title="GMAT BKT 模拟器", layout="wide")
st.title("GMAT BKT 自适应推荐题目演示版")

# 上区：参数设置（放在侧边栏）
st.sidebar.header("参数设置")
init_val = st.sidebar.slider("初始掌握度", 0.0, 1.0, 0.5, 0.05)
guess = st.sidebar.slider("猜测概率", 0.1, 0.5, 0.2, 0.05)
slip = st.sidebar.slider("失误概率", 0.0, 0.3, 0.1, 0.05)
learn = st.sidebar.slider("学习率", 0.0, 0.5, 0.1, 0.05)
threshold = st.sidebar.slider("掌握阈值", 0.0, 1.0, 0.6, 0.05)
prereq = st.sidebar.slider("先修要求", 0.0, 1.0, 0.7, 0.05)

# 下区：左下区操作区，右下区显示知识图谱与学习路径
col_left, col_right = st.columns([1, 2])

with col_left:
    st.header("操作区")
    if st.button("开始模拟"):
        st.session_state.sim = BKTSimulation(knowledge_points, num_questions, init_val, guess, slip, learn)
        st.session_state.sim.recommend_question(threshold, prereq)
        st.success("模拟已开始！")
    answer = st.radio("请选择当前题答案", ("Correct", "Incorrect"))
    if st.button("提交答案"):
        if "sim" in st.session_state and st.session_state.sim and st.session_state.sim.current_recommendation:
            correct = (answer == "Correct")
            st.session_state.sim.submit_answer(correct)
            st.session_state.sim.recommend_question(threshold, prereq)
            st.success("答案已提交！")
    if st.button("撤销更新"):
        if "sim" in st.session_state and st.session_state.sim:
            st.session_state.sim.undo_last_update()
            st.success("上一次更新已撤销！")
    st.markdown("---")
    if "sim" in st.session_state and st.session_state.sim and st.session_state.sim.current_recommendation:
        kp, qnum = st.session_state.sim.current_recommendation
        st.markdown(f"**【自动推荐】下一题：** 知识点 **{kp}**，题号：**{qnum}**")
    else:
        st.markdown("**等待推荐...**")

with col_right:
    st.header("知识图谱与学习路径")
    if "sim" in st.session_state and st.session_state.sim:
        html_graph = st.session_state.sim.get_pyvis_graph_html()
        st.components.v1.html(html_graph, height=600, scrolling=True)
        st.subheader("学习路径")
        st.write(st.session_state.sim.get_learning_path_str())
    else:
        st.info("请点击左侧【开始模拟】按钮开始。")
