from rbmBase import *
import os
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from Dataset import SetGenerator
import sys
import time

"""
calculate true, false postive and negative and Precision,sensitivity, specificity for each label and accuracy combined
"""
def calculteSVM_Results(SVM_pred,real_names,labelSet):
    top=Toplevel()
    guiStatsFrame = Frame(top)
    top.geometry("1000x250")
    TP = np.zeros(len(labelSet), dtype = int) 
    TN = np.zeros(len(labelSet), dtype = int) 
    FP = np.zeros(len(labelSet), dtype = int) 
    FN = np.zeros(len(labelSet), dtype = int) 
    guiLabels = []

    index = 0
    for label in labelSet:
        for i, point in enumerate(SVM_pred):
            if SVM_pred[i]!=label:
                if SVM_pred[i] == real_names[i]:
                    TN[index] =TN[index] + 1
                else:
                    FN[index] = FN[index] + 1 
            else:
                if SVM_pred[i] == real_names[i]:
                    TP[index] =TP[index] + 1
                else:
                    FP[index] = FP[index] + 1 
        index += 1
        
    txt = ""
    index = 0
    for label in labelSet:
        p= TP[index]/(TP[index]+FP[index])
        r= TP[index]/(TP[index]+FN[index])
        specificity=1-(TN[index]/(TN[index]+FP[index]))
        myStr = "Label: {}\nPrecision: {:.2f}\nSensitivity: {:.2f}\nSpecificity: {:.2f}\nTP: {},FP: {},FN: {},TN:{}\n\n".format(label,p,r,specificity,TP[index],FP[index],FN[index],TN[index])
        txt += myStr
        index += 1
        tempLabel = Label(guiStatsFrame,text=myStr,padx=20,pady=20)
        guiLabels.append(tempLabel)
        
    for i,label in enumerate(guiLabels):
        label.pack(side = LEFT)
    
    accuracy = (TP[0] +TN[0])/(TP[0] + TN[0] + FP[0] +FN[0])

    label2=Label(guiStatsFrame,text="SVM Accuracy:{:.2f}".format(accuracy)).pack(side = BOTTOM)
    guiStatsFrame.pack()

approved_file_types=[".gif"]
"""GUI Logic"""
class ProgController():
    def __init__(self,root,frameTitleList,approved_file_types):
        self.root=root
        self.approved_file_types=approved_file_types
        self.frameTitleList=frameTitleList
        self.index=0
        self.textbox = Text(root)
        
    def next_index(self):
        if self.index < len(self.frameTitleList):
            self.index=self.index +1
        return self.index
    def back_index(self):
        if self.index!=0:
            self.index=self.index -1
        return self.index
    def redirector(self,inputStr):
        self.textbox.insert(INSERT, inputStr)
    def training(self,localFrame):
        processLabel=Label(localFrame,text="processing...").pack()
        self.root.update()
        self.textbox.pack()
        localFrame.destroy()

      
        self.rbm.train(self.dataGenerator, 10)
        
        pics_probs=[]
        labels=[]
        training_set =self.dataGenerator.get_training_set()
        count=0
        for item in training_set:
            count+=1
            pic_prob=np.reshape(item[0],(1,item[0].shape[0]))
            pic_prob,_= self.rbm.sample_hidden(torch.from_numpy(pic_prob))
            labels.append(item[1])
            pic_prob=np.reshape(pic_prob.numpy(),pic_prob.shape[1])
            pics_probs.append(pic_prob)
        
        pics_probs =np.asarray(pics_probs)
        
        self.labelSet = []
        [self.labelSet.append(x) for x in labels if x not in self.labelSet] 
        labels=np.array(labels)
        
        from sklearn.svm import SVC
        svclassifier = SVC(degree=8, kernel='poly')
        
        
        col_bool=[]
        variance=[]
        for i in range(len(pics_probs[1])):
            variance.append(np.var(pics_probs[:,i]))
        
        for i in range(10):
            index = np.argmax(variance)
            variance[index]=0
            col_bool.append(index)
            
        clean_data=[]
        for i in range(len(pics_probs)):
            pic_data=[]
            for col in col_bool:
                pic_data.append(pics_probs[i,col])
              
            clean_data.append(pic_data)
            
        clean_data=np.array(clean_data)
        
        self.clean_data=clean_data
        self.labels=labels
        
        svclassifier.fit(clean_data, labels)
        self.predictions= svclassifier.predict(self.clean_data)
     
    def loadNextStage(self,index):
        mainFrame= Frame(self.root)
        self.next_index()
        if index==0 :
            dataset_frame=DatasetFrame(self, mainFrame, self.frameTitleList[self.index], self.index)
            dataset_frame.lables_field_label=Label(dataset_frame.frame,text="prefix").grid(rows=1, column=1,padx=10)
            dataset_frame.lables_field=Entry(dataset_frame.frame).grid(rows=2,column=1)
            dataset_frame.browse_btn=Button(dataset_frame.frame,text="choose dataset folder",command=dataset_frame.getDatasetDirectory)
            dataset_frame.browse_btn.grid(rows=2,column=2)
        elif index==1:
            training_frame=FramesAfterLoadingData(self, mainFrame, self.frameTitleList[self.index], self.index)

            trainBtn=Button(training_frame.frame,text="train",command=lambda:self.training(training_frame.frame)).pack()
            
        

        elif index ==2:
            results_frame=FramesAfterLoadingData(self, mainFrame, self.frameTitleList[self.index], self.index)
        
            deleteDataBtn=Button(results_frame.frame,text="Delete Datasets",command=self.dataGenerator.deleteSets).grid(rows=1,column=1)
            from rbmBaseold import printScatterTrainPlot
            graphBtn=Button(results_frame.frame,text="View Graph",command=lambda:printScatterTrainPlot(self.clean_data,self.labels)).grid(rows=1,column=2)
            stats_btn=Button(results_frame.frame,text="stats",command=lambda:calculteSVM_Results(self.predictions,self.labels, self.labelSet)).grid(rows=2, column=1)
        
          
            
    def startProg(self):
         mainFrame= Frame(self.root)
         startFrame=ProgFrame(self,mainFrame, self.frameTitleList[0],0)
         welcomeLabel=Label(startFrame.frame,bg="white",text="Welcome To RBM Picture Modeling\n\n\n\n\nCreated by:Ira Goor,Chen Hess,Eden Schwartz And Boaz Trauthwein").pack()
         mainFrame.place(anchor="center",relx=.5,rely=.5)
         mainFrame.configure(bg="white")
    
            
            
        
""""General frame base controller """
class ProgFrame():
    def __init__(self,controller,mainFrame,title,stage_index):
        
        self.controller=controller
        self.mainFrame=mainFrame
        self.frame=LabelFrame(self.mainFrame,text=title,bg="white")
        self.stage_index=stage_index
        self.exit_frame=LabelFrame(self.mainFrame,text='Exit Program',bg="white")
        self.exit_btn=Button(self.exit_frame,text="EXIT",command=self.exitProg)
        self.frame.pack(side=TOP)
        self.exit_btn.pack(side=RIGHT)
        self.stage_scroller=LabelFrame(self.mainFrame,text='Stages',bg="white")
        self.stage_scroller.pack(side=BOTTOM)
        if self.stage_index!=5:
            self.next_btn=Button(self.stage_scroller,text="Next",bg="white",command=self.nextStage,padx=40,pady=20)
            self.next_btn.pack(side=RIGHT)
       
        mainFrame.pack()
        
        
        
    def exitProg(self):
        self.controller.root.close()
    def lastStage(self):
        pass

        
    def nextStage(self):
        if self.checkCorrectness():
            self.mainFrame.destroy()
            self.controller.loadNextStage(self.stage_index)
        return
        
        
    def checkCorrectness(self):
        return True
    
""""Initialize the experiment data, frame controller    """
class DatasetFrame(ProgFrame):


   
    def getDatasetDirectory(self):

        self.directory= filedialog.askdirectory()

        
    def checkCorrectness(self):
        
        prefix="subject"
        dirs = os.listdir(self.directory )
        for data in dirs:
            if os.path.isfile(self.directory+ "\\" + data):
                if (data.find(prefix)==-1):
                    messagebox.showerror("Dataset Error","File must have only files from an aprroved prefix")
                    return False
            f, e = os.path.splitext(self.directory + "\\" + data)
            if e not in self.controller.approved_file_types:
                messagebox.showerror("Dataset Error","File must have only files from an aprroved format")
                messagebox.showinfo("files approved:" + str(self.controller.approved_file_types) + '\nchoose again')
                return False
        return True

        
    def nextStage(self):
        if self.checkCorrectness():
            h=100
            w=100
            self.controller.dataGenerator=SetGenerator(self.directory,(h,w),0.2)
            self.controller.rbm=RBM(hidden_size = 40, n_epoch=30,visible_size=(h*w),batch_size=10)
        super().nextStage()


""""General frame for post data initilialization """
class FramesAfterLoadingData(ProgFrame):
      def exitProg(self):
          self.controller.dataGenerator.deleteSets()
          self.controller.root.quit()
          self.controller.root.destroy()
          
        
"""MAIN"""
root=Tk()
root.geometry("800x600")
root.configure(bg="#E2E8A8")

progController=ProgController(root, ["intro","dataset Loading","training","results"], approved_file_types)
sys.stdout.write = progController.redirector #whenever sys.stdout.write is called, redirector is called.

progController.startProg()
root.mainloop()






