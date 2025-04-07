#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install groq


# In[2]:


pip install pandas


# In[10]:


import pandas as pd
from groq import Groq
import csv
import re

API_KEY = "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"
client = Groq(api_key=API_KEY)

submit = pd.read_csv("mmlu_submit.csv")
submit = submit.rename(columns={"Unnamed: 0": "ID"})
print("Question 數量：", len(submit))

prompt = """
You are a concise expert in {task} questions, focused on quickly and accurately answering multiple-choice questions by providing only the answer (A, B, C, or D) without explanation or extra text in the final output. Your task is to read the question and options below and determine the correct answer using a structured reasoning process.

Before answering:
1. Read the question closely, identifying key terms, requirements, and complexity level.
2. Use a step-by-step Chain of Thought (CoT) approach internally:
   - Break the question into logical parts based on its subject (e.g., for math, factor equations; for geography, recall facts; for history, align dates/events).
   - Evaluate each option systematically, eliminating incorrect ones using subject-specific knowledge.
   - If ambiguity or potential error is detected, revisit the question and options to resolve it.
   - After selecting an answer, critique it: Does it fully satisfy the question? If not, re-evaluate.
3. Adjust your analytical approach based on the subject (e.g., calculations for mathematics, factual recall for history, exclusion for negative questions).
4. Double-check the final choice by reapplying the question’s criteria.
5. Verify the answer:
   - For math: Substitute the chosen value back into the problem.
   - For facts: Ensure alignment with established knowledge.
   - For negative questions: Confirm all other options meet the positive condition.
6. Respond with ONLY the letter of the correct answer (A, B, C, or D).

Example 1 (Mathematics - Quadratic Equation):  
Question: If x^2 - 5x + 6 = 0, which of the following is a possible value of x?  
Options: A: 1  B: 2  C: 3  D: 4  
Answer: B  

Example 2 (Geography - Desert Size):  
Question: On which continent is the largest desert by area located?  
Options: A: Africa  B: Asia  C: Australia  D: Antarctica  
Answer: D  

Example 3 (History - Treaty Context):  
Question: The Treaty of Versailles, signed in 1919, primarily ended which conflict?  
Options: A: World War II  B: The Napoleonic Wars  C: World War I  D: The Franco-Prussian War  
Answer: C  

Example 4 (Science - Elemental Discovery):  
Question: Which element was discovered on the Sun before Earth and is the second most abundant element in the universe?  
Options: A: Hydrogen  B: Helium  C: Oxygen  D: Carbon  
Answer: B  

Example 5 (Negative Question - Biology):  
Question: Which of the following is not a mammal?  
Options: A: Dolphin  B: Crocodile  C: Whale  D: Bat  
Answer: B  

Example 6 (Mathematics - Systems of Equations):  
Question: If 2x + y = 5 and x - y = 1, what is the value of x?  
Options: A: 1  B: 2  C: 3  D: 4  
Answer: C  

Example 7 (Geography - River Systems):  
Question: Which river, flowing through multiple countries, is the longest in South America and the second longest in the world by discharge volume?  
Options: A: Nile  B: Amazon  C: Paraná  D: Orinoco  
Answer: B  

Example 8 (History - Cold War):  
Question: Which event, occurring in 1962, brought the United States and the Soviet Union closest to nuclear war?  
Options: A: Berlin Wall Construction  B: Bay of Pigs Invasion  C: Cuban Missile Crisis  D: Korean War  
Answer: C  

Example 9 (Science - Physics):  
Question: If an object is thrown upward at 20 m/s on Earth (ignoring air resistance), how long does it take to reach its maximum height? (Use g = 10 m/s²)  
Options: A: 1 s  B: 2 s  C: 3 s  D: 4 s  
Answer: B  

Example 10 (Logic - Negative Multi-Step):  
Question: Which of these is not a prime number greater than 10?  
Options: A: 11  B: 13  C: 15  D: 17  
Answer: C

Now, answer the following {task} question:
Question: {question}
Options:
{options}

Answer (A, B, C, or D):
"""

def get_groq_response(question, options, task):
    formatted_prompt = prompt.format(
        task = task,
        question=question,
        options="  ".join([f"{key}: {value}" for key, value in options.items()]) #options 轉換為 "A:... B:... C:... D:..." 的格式
    )
    result = client.chat.completions.create(
        messages=[{"role": "user", "content": formatted_prompt}],
        model="deepseek-r1-distill-llama-70b",
        timeout=60
    )
    response = result.choices[0].message.content.strip() #取出AI回答
    print(f"問題：{question} | 模型回應：{response}")  # 看模型的回答
        
    # 抓回應中最後的單一字母
    match = re.search(r'[ABCD]$', response) #re取'^[ABCD]$'的話會有問題
    return match.group(0) if match else "C" #若AI沒有答案的話猜C
    

# 處理、預測答案
results = []
for index, row in submit.iterrows():
    question = row["input"]
    options = {
        "A": row["A"],
        "B": row["B"],
        "C": row["C"],
        "D": row["D"]
    }
    task = row["task"]
    predicted_answer = get_groq_response(question, options, task)
    question_id = row["ID"]
    results.append([question_id, predicted_answer])
    print(f"已解決Question ID： {question_id}")

# 儲存csv
output_file = "submit.csv"
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "target"])
    writer.writerows(results)


# In[7]:


pip freeze | grep -E 'pandas|groq'


# In[ ]:




