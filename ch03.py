 
# 编程实践， 动态规划求解小型方格世界最优策略


S=[i for i in range(16)]# 状态空间
A=["n","e","s","w"] #行为空间
# P,R,将由dynamics动态生成



 #行为对状态的改变,
 #例如,在状态8,执行行为n(北,向上) ,将改变状态为 4  ,即 8 + (-4) = 4
ds_actions = {"n":-4,"e":1 ,"s":4 ,"w":-1}

def dynamics(s,a):# 环境动力学
	'''模拟小型方格世界的环境动力学特征
	Args:
		s 当前状态 int 0 - 15
		a 行为 str in ['n','e','s','w'] 分别表示 北, 东,南,西 
	Returns : tuple(s_prime, reward ,is_end)
		s_prime 后续状态
		reward 奖励值
		is_end 是否进入终止状态
	'''
	s_prime=s
	if (s%4 == 0 and a == 'w') or (s<4 and a =='n') or \
		((s+1)%4 == 0 and a == 'e') or (s>11 and a == 's') or \
		s in [0,15]:
		# 边界
		pass
	else:
		ds = ds_actions[a]
		s_prime = s+ds

	reward = 0 if s in [0,15] else -1
	is_end = True if s in [0,15] else False
	return s_prime, reward ,is_end

def P(s,a,s1): #状态转移概率函数
	s_prime, _ ,_ = dynamics(s, a)
	return s1 == s_prime

def R(s,a):#奖励函数
	_,r,_ = dynamics(s,a)
	return r

gamma = 1.00
MDP = S,A,R,P ,gamma # MDP拥有五个元素的元组, 只不过R和P都变成了函数， 而不是字典


#均一随机策略
def uniform_random_pi(MDP=None, V=None, s=None, a=None):
	'''均一随机策略,只需要知道行为空间
	'''
	_,A,_,_,_ = MDP
	n=len(A)
	return 0 if n == 0 else 1.0/n

#贪婪策略
def greedy_pi(MDP,V,s,a):
	'''贪婪策略
		需要知道状态的价值,
		考虑了多个状态具有相同最大值的情况， 此时贪婪策略从这多个具有相同最大值的行为中，随机选择一个
	'''
	S,A,P,R gamma = MDP
	max_v, a_max_v = -float('inf'),[]

	# 统计后续状态的最大价值以及到达该状态的行为(可能不止一个)
	for a_opt in A:
		s_prime,reward,_ = dynamics(s,a_opt)
		v_s_prime = get_value(V,s_prime)
		if v_s_prime > max_v:
			max_v = v_s_prime
			a_max_v = [a_opt]
		elif( v_s_prime == max_v):
			a_max_v.append(a_opt)


	n = len(a_max_v)
	if n == 0:return 0.0
	return 1.0/n if a in a_max_v else 0.0


def get_pi(Pi,s,a,MDP=None,V=None):
	return Pi(MDP,V,s,a)


#辅助函数
def get_prob(P,s,a,s1):#获取状态转移概率
	return P(s,a,s1)

def get_reward(R,s,a):#获取奖励值
	return R(s,a)

def set_value(V,s,v):#设置价值字典
	V[s]= v

def get_value(V,s):#获取状态价值
	return V[s]

def display_V(V):#显示状态价值
	for i in range(16):
		print('{0:>6.2f}'.format(V[i]),end=" ")
		if(i+1) %4 == 0:
			print("")
	print()



def compute_q(MDP,V,s,a):
	'''根据给定的MDP,价值函数V, 计算状态行为对(s,a)的价值qsa
	'''
	S,A,R,P ,gamma = MDP
	q_sa = 0
	for s_prime in S:
		q_sa += get_prob(P,s,a,s_prime) * get_value(V,s_prime)

	q_sa = get_reward(R,s,a) + gamma * q_sa
	return q_sa

def compute_v(MDP,V,Pi,s):
	'''给定MDP下依据某一策略Pi和当前状态价值函数V计算某状态s的价值
	'''
	S,A,R,P ,gamma = MDP
	v_s = 0
	for a in A:
		v_s += get_pi(Pi,s,a,MDP,V) * compute_q(MDP,V,s,a)
	return v_s


def update_v(MDP,V,Pi):
	'''给定一个MDP和一个策略， 更新该策略下的价值函数V
	'''
	S,_,_,_,_ = MDP
	V_prime = V.copy()
	for s in S:
		set_value(v_prime, s, compute_v(MDP, v_prime,Pi,s))
	return V_prime


def policy_evaluate(MDP,V,Pi,n):
	'''策略评估: 使用n次迭代计算来评估一个MDP在给定策略Pi下的状态价值， 初始时， 价值为V
	'''
	for i in range(n):
		V=update_V(MDP,V,Pi)
	return V

def policy_iterate(MDP,V,Pi,n,m):
	for i in range(m):
		V=policy_evaluate(MDP,V,Pi,n)
		Pi = greedy_pi #第一次迭代产生新的价值函数后，随机使用贪婪策略
	return V