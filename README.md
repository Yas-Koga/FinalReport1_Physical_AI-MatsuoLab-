# FinalReport1_Physical_AI-MatsuoLab-
# 松尾研フィジカルAI応用編１の最終課題のリポジトリです。
Took ten episodes by crane V2 teleop

To deploy 

pixi run infer   --model-path /home/ubuntu/checkpoints/train/crane_plus_pekori_act   --joint-limits joint_limits.json   --enable-publish   --confirm-dataset-validated


Joint limitations for Gazebo(Servo_Home 150degree)

Joint name, Degree, Radian

1L	0	    -2.617993878

1U	300	    2.617993878

2L	45.45	-1.824741733

2U	252.2	1.783726496

3L	3.52	-2.556558288

3U	290.62	2.454281994

4L	44.57	-1.84010063

4U	251.32	1.768367598

HL	109.38	-0.708952742

HU	188.27	0.667937505