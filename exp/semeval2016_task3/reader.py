import xml.etree.ElementTree as ET


DEV=['./data/v3.2/dev/SemEval2016-Task3-CQA-QL-dev.xml']
TRAIN=['./data/v3.2/train/SemEval2016-Task3-CQA-QL-train-part1.xml','./data/v3.2/train/SemEval2016-Task3-CQA-QL-train-part2.xml']
TEST=['./data/SemEval2016_task3_test_input/English/SemEval2016-Task3-CQA-QL-test-input.xml']
EXTRA=['./data/v3.2/train-more-for-subtaskA-from-2015/SemEval2015-Task3-CQA-QL-train-reformatted-excluding-2016-questions-cleansed.xml',
       './data/v3.2/train-more-for-subtaskA-from-2015/SemEval2015-Task3-CQA-QL-dev-reformatted-excluding-2016-questions-cleansed.xml',
       './data/v3.2/train-more-for-subtaskA-from-2015/SemEval2015-Task3-CQA-QL-test-reformatted-excluding-2016-questions-cleansed.xml']

'''
data=[
[ori_q.attrib, [ori_q.subject,ori_q.body], rel_q.attrib, [rel_q.subject,rel_q.body]... 10 relevent questions]
...
]
'''

class Reader:
    def __init__(self):
        pass

    def getText(self, fnames):
        data=[]
        for i in range(len(fnames)):
            _data = self.loadText(fnames[i])
            data=data+_data
        return data

    def loadText(self, fname):
        data=[]
        tree = ET.parse(fname)
        for elem in tree.iter():
            if elem.text!=None and elem.text!='':
                text=elem.text
                text=text.rstrip('\n')
                text=text.rstrip('\t')
                text=text.rstrip('\n\t')
                if text!='':
                    data.append(text)
        return data

    def getData(self,fnames):
        data=[]
        for i in range(len(fnames)):
            data = self.load(data,fnames[i])
        return data

    def load(self, data, fname):
        tree = ET.parse(fname)
        root = tree.getroot()


        count=0
        cur_ori_q=None
        for ori_q in root:
            if cur_ori_q is None or ori_q.attrib['ORGQ_ID']!=cur_ori_q:
                data.append([ori_q.attrib])
            for ori_q_c in ori_q:
                if ori_q_c.tag=='OrgQSubject':
                    if cur_ori_q is None or ori_q.attrib['ORGQ_ID']!=cur_ori_q:
                        data[-1].append([ori_q_c.text])
                elif ori_q_c.tag=='OrgQBody':
                    if cur_ori_q is None or ori_q.attrib['ORGQ_ID']!=cur_ori_q:
                        data[-1][-1].append(ori_q_c.text)
                elif ori_q_c.tag=='Thread':
                    for thread_c in ori_q_c:
                        if thread_c.tag=='RelQuestion':
                            data[-1].append(thread_c.attrib)
                            for relQuestion_c in thread_c:
                                if relQuestion_c.tag=='RelQSubject':
                                    data[-1].append([relQuestion_c.text])
                                elif relQuestion_c.tag=='RelQBody':
                                    if relQuestion_c.text is None:
                                        data[-1][-1].append(' ')
                                    else:
                                        data[-1][-1].append(relQuestion_c.text)
            cur_ori_q=ori_q.attrib['ORGQ_ID']
            count+=1
        return data


