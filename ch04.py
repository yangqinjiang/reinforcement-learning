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
		
		self.name = name
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