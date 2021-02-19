# -*- coding: utf-8 -*-
# @Time : 2021/2/19 14:57
# @Author : Jclian91
# @File : cpm_usage_demo.py
# @Place : Yangpu, Shanghai
import tensorflow_hub as hub
import tensorflow as tf

from gpt2_tokenizer import GPT2Tokenizer

tokenizer = GPT2Tokenizer(
    'CPM-Generate/bpe_3w_new/vocab.json',
    'CPM-Generate/bpe_3w_new/merges.txt',
    model_file='CPM-Generate/bpe_3w_new/chinese_vocab.model')

gpt = hub.load('/nlp_group/nlp_pretrain_models/cpm-lm-tf2_v2/')


def sample(tokenizer, gpt, sentence, number=1, length=20, top_p=0.9, temperature=0.9):
    """
    numbert: 输出句子个数
    length: 输出最大长度
    top_p: token的概率排在这以上才有效
    temperature: 温度
    """
    inputs = tf.constant([tokenizer.encode(sentence)] * number, dtype=tf.int64)
    length = tf.constant(length, dtype=tf.int64)
    ret = gpt.signatures['serving_default'](
        inp=inputs,
        length=length,
        top_p=tf.constant(top_p, tf.float32),
        temperature=tf.constant(temperature, tf.float32)
    )['output_0']
    return [
        tokenizer.decode(s).replace(' ', '')
        for s in ret.numpy()
    ]


# 英语默写
ret = sample(tokenizer, gpt, '默写英文：\n狗->dog\n猫->cat\n鸟->bird\n猪->pig\n鱼->fish\n羊->', 1, 12, top_p=0.9, temperature=0.9)
for x in ret:
    print(x)
    print('-' * 20)

"""
输出结果:(每次输出的结果可能会不一致)
默写英文:
狗->dog
猫->cat
鸟->bird
猪->pig
鱼->fish
羊->sheep
鸭子->duck

--------------------
"""

# 常识推理
query = """
美国的首都是华盛顿\n
法国的首都是巴黎\n
日本的首都是东京\n
中国的首都是
"""
ret = sample(tokenizer, gpt, query, 1, 10, top_p=0.9, temperature=0.9)
for x in ret:
    a = x[len(query):]
    answer = a.split('\n')[0]
    print("answer:", answer)
    print('-' * 20)

"""
输出结果:
answer: 北京
--------------------
"""

query2 = """
南开大学位于天津\n
清华大学在北京\n
南京大学属于南京\n
中山大学在广州\n
复旦大学位于
"""
ret = sample(tokenizer, gpt, query2, 1, 10, top_p=0.9, temperature=0.9)
for x in ret:
    a = x[len(query2):]
    answer = a.split('\n')[0]
    print("answer:", answer)
    print('-' * 20)

"""
输出结果:
answer: 上海
--------------------
"""


# 简易问答
def ask_gpt(question):
    q = f'''
问题：中国首都是哪里？
答案：北京
问题：世界上国土面积最大的国家是哪个？
答案：俄罗斯
问题：李白生活在哪个朝代？
答案：唐朝
问题：美国最大的城市是哪座？
答案：纽约
问题：{question}
答案：
'''
    ret = sample(tokenizer, gpt, q, 3, 10, top_p=0.9, temperature=0.9)
    answers = []
    for x in ret:
        a = x[len(q):]
        a = a.split('\n')[0]
        answers.append(a)
    return answers


print(ask_gpt('张居正是哪个朝代的人？'))
# ['(15)、(16)', '张居正', '问题:你印象最深的历史人物是谁']
print(ask_gpt('世界上最深的湖泊是哪个？'))
# ['特拉华河', '[6]', '问:有多少个省?']
print(ask_gpt('李世民的父亲是谁？'))
# ['李渊', '李渊', '唐太宗']
print(ask_gpt('python和java哪个难学？'))
# ['⁇————', '答案是:python', 'python']
print(ask_gpt("上海一共有几个区？"))
# ['1、黄浦区', '上海浦东新区', '问题:世界上海拔最高的山峰是']


# 文本扩写
ret = sample(tokenizer, gpt, '没有梦想的人，', 3, 50, top_p=0.9, temperature=0.9)
for x in ret:
    print(x)
    print('-' * 20)

"""
扩写结果:
没有梦想的人,就去找自己的梦想。不要把自己的快乐建立在别人的痛苦之上。你的同学告诉你的是他的一面之词,你自己在这方面已经有自己的判断了。
--------------------
没有梦想的人,那就好好享受属于自己的那份快乐。谁说大学生就业难,没有一个数据支持,让大家都心服口服。但是,我们有数据可以拿出来,那
--------------------
没有梦想的人,就像庄子所说的,“圣人不死,大盗不止”。只有真正具备梦想的人,才能做到既不被羁绊,又能实现自己
--------------------
"""

ret = sample(
    tokenizer, gpt,
    '一时黛玉进了荣府，下了车。众嬷嬷引着，便往东转弯，穿过一个东西的穿堂，向南大厅之后，仪门内大院落，上面五间大正房，两边厢房鹿顶耳房钻山，四通八达，轩昂壮丽，比贾母处不同。黛玉便知这方是正经正内室，一条大甬路，直接出大门的。',
    3, 200, top_p=0.9, temperature=0.9)

for x in ret:
    print(x)
    print('-' * 20)

"""
扩写结果:
一时黛玉进了荣府,下了车。众嬷嬷引着,便往东转弯,穿过一个东西的穿堂,向南大厅之后,仪门内大院落,上面五间大正房,两边厢房鹿顶耳房钻山,四通八达,轩昂壮丽,比贾母处不同。黛玉便知这方是正经正内室,一条大甬路,直接出大门的。因笑道:“方才的喜鹊是个什么?”袭人道:“是个小耗,大耗子。”黛玉笑道:“我看是耗子偷吃了果子拉的屎。”袭人道:“那原不算什么,只是耗子喜欢吃屎。”黛玉笑道:“你这么大个人,怎么就知道他是耗子呢?”袭人道:“他比耗子大得多,就比耗子大,又比耗子本事大。”黛玉笑道:“话是这么说,人还未见,人先见了耗子,岂不是见了耗子的造化了?”袭人道
--------------------
一时黛玉进了荣府,下了车。众嬷嬷引着,便往东转弯,穿过一个东西的穿堂,向南大厅之后,仪门内大院落,上面五间大正房,两边厢房鹿顶耳房钻山,四通八达,轩昂壮丽,比贾母处不同。黛玉便知这方是正经正内室,一条大甬路,直接出大门的。但见院子里梧桐森森,⁇珞争辉,只是不多,人已稀了。刚进了院子,黛玉乍见正面墙上垂着一条“⁇芜苑”三个大黑字,又见一个纱橱,橱内露出纱衣半截,纱橱上并无“⁇芜苑”三个字,正疑惑间,忽听见有人笑声,不觉站住,也不辨是谁,仔细看时,只见一个人,蓬头垢面,身上都是破烂之物,赤着一双足,向黛玉扑来。黛玉真不料是此人,连忙躲闪。那人却也不惧怕,口中唱
--------------------
一时黛玉进了荣府,下了车。众嬷嬷引着,便往东转弯,穿过一个东西的穿堂,向南大厅之后,仪门内大院落,上面五间大正房,两边厢房鹿顶耳房钻山,四通八达,轩昂壮丽,比贾母处不同。黛玉便知这方是正经正内室,一条大甬路,直接出大门的。黛玉回头看时,只见院子里甚是清静,只听见小丫头们说话。因问黛玉:“有什么事?”黛玉道:“园内的花儿真好看,今儿一早,不知那一处的花儿先开了。”黛玉又道:“那是昨儿林姑娘送过来的。”一面说,一面走,又被宝玉看见,说道:“头里去了,我就忘了。”黛玉道:“他们大概看着也好看,也就送过来了。”当黛玉走进大厅之后,宝玉早吓得脸白,大叫一声:“林姑娘救命!”飞身去救。这时
--------------------
"""

# 主语抽取
query = """
弗朗西斯·培根是英国唯物主义哲学家、思想家和科学家，被马克思称为“英国唯物主义和整个现代实验科学的真正始祖”。->弗朗西斯·培根
1715年，伏尔泰因写诗讽刺当时摄政王奥尔良公爵被流放到苏里。->伏尔泰
对这位未来的世界领袖，亚里士多德灌输了道德、政治以及哲学方面的知识，对亚历山大的思想形成起了重要的作用。->亚里士多德
1844年尼采出生于勒肯的一个牧师之家，他自幼性情孤僻，而且多愁善感，纤弱的身体使他总是有一种自卑感。->尼采
1889年4月26日，路德维希·约瑟夫·约翰·维特根斯坦出生于当时是奥匈帝国的维也纳。->
"""
ret = sample(tokenizer, gpt, query, 1, 20, top_p=0.9, temperature=0.9)
for x in ret:
    a = x[len(query):]
    answer = a.split('\n')[0]
    print("answer:", answer)
    print('-' * 20)

# 输出结果: 维特根斯坦

# 关系抽取
query = """
姚明的身高是211cm，是很多人心目中的偶像。 ->姚明，身高，211cm
虽然周杰伦在欧洲办的婚礼，但是他是土生土长的中国人->周杰伦，国籍，中国
小明出生于武汉，但是却不喜欢在武汉生成，长大后去了北京。->小明，出生地，武汉
吴亦凡是很多人的偶像，但是他却是加拿大人，另很多人失望->吴亦凡，国籍，加拿大
武耀的生日在5月8号，这一天，大家都为他庆祝了生日->武耀，生日，5月8号
《青花瓷》是周杰伦最得意的一首歌。->周杰伦，作品，《青花瓷》
北京是中国的首都。->中国，首都，北京
蒋碧的家乡在盘龙城，毕业后去了深圳工作。->蒋碧，籍贯，盘龙城
上周我们和王立一起去了他的家乡云南玩昨天才回到了武汉。->王立，籍贯，云南
昨天11月17号，我和朋友一起去了海底捞，期间服务员为我的朋友刘章庆祝了生日。->刘章，生日，11月17号
王红的体重达到了140斤，她很苦恼。->
"""
ret = sample(tokenizer, gpt, query, 1, 20, top_p=0.9, temperature=0.9)
for x in ret:
    a = x[len(query):]
    answer = a.split('\n')[0]
    print("answer:", answer)
    print('-' * 20)

# 输出结果: 红,体重,140斤