# Client(Franka)

```docker start keymanip ```

```docker exec -it keymanip /bin/bash```

```cd data/keymanip/task_planning```

```
python ws_client_final.py(在原子技能和server端启动之后再执行)
用于生成拆解任务并向原子技能端和server端传递
```





# Server(5090(kyf))

```docker start task_plan```

```docker exec -it task_plan /bin/bash```

```cd ../home/VLM_infer```

```python ws_server.py(beifen_server.py)```

```

ws_qwen_server.py用于qwen推理

prompt_dict_final.txt用于管理微调技能的技能名称和权重路径
```