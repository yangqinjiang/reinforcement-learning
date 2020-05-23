from random import shuffle
from queue import Queue
from tqdm import tqdm #进度条
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from utils import str_key,set_dict,get_dict

class Gamer():
	"""游戏者"""
	def __init__(self, name="",A=None,display=False):
		
		self.name = name  #游戏者的姓名
		self.cards = [] #手中的牌
		self.display = display #是否显示对局文字信息
		self.policy = None #策略
		self.learning_method = None #学习方法
		self.A = A #行为空间

	def __str__(self):
		return self.name

	def _value_of(self,card):
		'''根据牌的字符，判断牌的数值大小，A被输出为1 , JQK均为10, 其他按牌字符对应的数字取值
		Args:
			card: 牌面信息 str
		Return:
			牌的大小数值, int, A返回1
		'''
		try:
			v = int(card)
		except:
			if card == 'A':
				v=1
			elif card in ['J','Q','K']:
				v=10
			else:
				v=0
		finally:
			return v
	
	def get_points(self):
		'''统计一手牌分值,如果使用了A的1点,同时返回True
		Args:
			cards 庄家或玩家手中的牌 list ['A','10','3']
		Return:
			tuple (返回牌总点数， 是否使用了可复用Ace)
			例如['A','10','3'] 返回 (14, False)
				['A','10']    返回 (21, True)
		'''
		num_of_useable_ace = 0 #默认没有拿到Ace
		total_point = 0 #总值
		cards = self.cards
		if cards is None:
			return 0, False

		# 考虑可能出现多张A的情况
		for card in cards:
			v=self._value_of(card)
			if v == 1: #判断Ace,v变成11
				num_of_useable_ace += 1
				v = 11
			total_point += v

		# 如果： 总分值大于21, 而且拿到Ace
		# 则： 总分值减去(num_of_useable_ace)个 A, 即 减10得到1
		while total_point > 21 and num_of_useable_ace > 0:
			total_point -= 10
			num_of_useable_ace -= 1
		
		return total_point , bool(num_of_useable_ace)



	def receive(self,cards=[]):
		'''玩家获得一张或多张牌'''
		cards = list(cards)
		for card in cards:
			self.cards.append(card)

	def discharge_cards(self):
		'''玩家把手中的牌清空， 扔牌'''
		self.cards.clear()

	def cards_info(self):
		'''玩家手中牌的信息'''
		self._info("{}{}现在的牌:{}\n".format(self.role, self,self.cards))
	
	def _info(self,msg):
		if self.display:
			print(msg,end="")



class Dealer(Gamer):
	"""庄家"""
	def __init__(self, name="",A=None, display=False):
		super(Dealer, self).__init__(name,A,display) #调用父类的__init__
		self.role = "庄家" #角色
		self.policy =self.dealer_policy #庄家的策略

	def first_card_value(self):
		'''显示第一张明牌'''
		if self.cards is None or len(self.cards) == 0:
			return 0

		return self._value_of(self.cards[0])
	
	def dealer_policy(self,Dealer=None):
		'''庄家策略的细节'''
		action=""
		dealer_point , _ = self.get_points()
		if dealer_point >= 17:  #策略一,总分值大于等于 17,则停止叫牌
			action = self.A[1] #停止叫牌
		else:
			action = self.A[0]  #继续叫牌
		return action

class Player(Gamer):
	"""玩家"""
	def __init__(self, name="",A=None, display=False):
		super(Player, self).__init__(name, A, display)
		self.role = "玩家"
		self.policy = self.naive_policy

	def get_state(self,dealer):
		'''根据当前局面信息得到当前局面的状态， 为策略评估做准备
		Return:
			dealer_first_card_value： 庄家的第一张牌
			Player_points:  玩家总分值
			useable_ace: 是否有可用的Ace
		'''
		dealer_first_card_value = dealer.first_card_value()
		player_points,useable_ace = self.get_points()
		return dealer_first_card_value, player_points , useable_ace
	
	def get_state_name(self,dealer):
		'''返回状态的key'''
		return str_key(self.get_state(dealer))

	def naive_policy(self,dealer=None):
		'''指定其策略为最原始的策略(naive_policy),规定玩家只要点数小于20点，就会继续叫牌'''
		player_points , _ = self.get_points()
		if player_points < 20: # 如果玩家的总分值小于20, 则继续叫牌, 否则停止叫牌
			action = self.A[0]
		else:
			action = self.A[1]
		return action