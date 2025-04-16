import cv2
import nuke
import os
import threading
import gc 
import torch
from NukeSamurai.scripts.demo import main

nuke.tprint('Current thread : ' +str(threading.current_thread().name) )


class InputInfos :
    read = None
    path = None
    original_fps = None
    bits = None
    
    @classmethod
    def getInputInfos(cls):
        f = nuke.thisNode().dependencies()

        for i in f:
            cls.read = i

        # Get file path
        try :
            cls.path = cls.read.knob('file').getValue()
        except : 
            cls.path = ''

        # Get metadatas
        ## get input fps
        try :
            cls.original_fps = int(cls.read.metadata()['input/frame_rate'])  
        except :
            cls.original_fps = nuke.Root()['fps'].value()  

        ## Get input bit depth
        # Look for bitsperchannel into the metadatas 
        try : 
            cls.bits = cls.read.metadata()['input/bitsperchannel']   
        except :
            # Interpreting bit depth depending on the format
            nuke.tprint("No Bit Depth information in the input metadatas. Interpreting it from the input extension.")

            if cls.path.endswith("jpg") or cls.path.endswith("jpeg") or cls.path.endswith("png") or cls.path.endswith("tiff"):
                cls.bits = "8-bit fixed"
            elif cls.path.endswith("exr") :
                cls.bits = "32-bit float"
            else :
                TypeError("Cannot interpret bit depth from unsupported input format.")
       

def UpdatePath():
    InputInfos.getInputInfos()
    FilePath = InputInfos.path
    nuke.thisNode().knob('FilePath').setValue(FilePath)
    

def GenerateMask():
    Output_path = nuke.thisNode().knob('OutputPath').getValue()

    # Checks           
    if str(os.path.splitext(os.path.basename(Output_path))[0]) == '' :
        raise TypeError("You must assign a file name")
        
    if nuke.thisNode()['FilePath'].value().lower().endswith("mp4") :
        raise TypeError('Unsupported input format. Input must be an Image Sequence')
        
    if nuke.thisNode().knob('FileType').value() == "exr" :
        if ("%04d" not in Output_path)  :        
            if ("%03d"not in Output_path)  :       
                raise TypeError("Your file must contains '####' or '###'")
        

    video_path= nuke.thisNode().knob('FilePath').value()
    video_output_path=nuke.thisNode().knob('OutputPath').getValue()
    save_to_file=nuke.thisNode().knob('FileType').value()
    # Get bbox coordinates from the node knobs
    bbox_x = nuke.thisNode().knob('BBoxX').value()
    bbox_y = nuke.thisNode().knob('BBoxY').value()
    bbox_w = nuke.thisNode().knob('BBoxW').value()
    bbox_h = nuke.thisNode().knob('BBoxH').value()
    bbox_coord = (int(bbox_x), int(bbox_y), int(bbox_w), int(bbox_h))
    
    frame_range = [int(nuke.thisNode().knob('FrameRangeMin').value()) , int((nuke.thisNode().knob('FrameRangeMax').value()+ 1))]
    original_fps = int(InputInfos.original_fps)                        
    target_fps=int(nuke.thisNode().knob('FPS').value())
    bits = InputInfos.bits

    if nuke.thisNode().knob('ModelType').value() == 'Large' :
        model_path = "sam2_repo/checkpoints/sam2.1_hiera_large.pt"
    if nuke.thisNode().knob('ModelType').value() == 'Base+' :    
        model_path = "sam2_repo/checkpoints/sam2.1_hiera_base_plus.pt"
    if nuke.thisNode().knob('ModelType').value() == 'Small' :    
        model_path = "sam2_repo/checkpoints/sam2.1_hiera_small.pt"
    if nuke.thisNode().knob('ModelType').value() == 'Tiny' :    
        model_path = "sam2_repo/checkpoints/sam2.1_hiera_tiny.pt"

    # Define childThread
    childThread = threading.Thread(target=main, args=( 
        video_path,
        video_output_path,
        bbox_coord, 
        save_to_file,
        frame_range,
        original_fps,                        
        target_fps,
        bits,
        model_path
    ))
    # Starting thread
    childThread.start()
    
    gc.collect()
    torch.cuda.empty_cache()


def CreateSamuraiNode():

    # Creating node
    nuke.createNode('NoOp')
    s = nuke.selectedNode()

    # Adding knobs
    s.knob('name').setValue('SAMURAI')
    s.addKnob(nuke.File_Knob('FilePath', 'File Path'))
    s.addKnob(nuke.PyScript_Knob('UpdatePath', 'Update Path', 'UpdatePath()' ))
    
    # Add Bounding Box knobs
    s.addKnob(nuke.Int_Knob('BBoxX', 'BBox X'))
    s.addKnob(nuke.Int_Knob('BBoxY', 'BBox Y'))
    s.addKnob(nuke.Int_Knob('BBoxW', 'BBox W'))
    s.addKnob(nuke.Int_Knob('BBoxH', 'BBox H'))
   
    s.addKnob(nuke.Text_Knob(''))

    s.addKnob(nuke.Int_Knob("FPS", 'Output Frame Rate'))
    s.addKnob(nuke.Int_Knob("FrameRangeMin", 'Frame Range'))
    s.addKnob(nuke.Int_Knob("FrameRangeMax", ' '))
    s.addKnob(nuke.Enumeration_Knob('ModelType', 'Model type', ['Base+','Large', 'Small', 'Tiny']))

    s.addKnob(nuke.Text_Knob(' ', ''))
    
    s.addKnob(nuke.Enumeration_Knob('FileType', 'File type', ['exr', 'mp4']))
    s.addKnob(nuke.File_Knob('OutputPath', 'Output Path'))
    s.addKnob(nuke.PyScript_Knob('GenerateMask', 'Generate Mask', 'GenerateMask()'))
    
    
    
    
### SETTING RANGES, DEFAULT VALUES, TOOLTIPS & FORMAT ###
    s['FPS'].setValue(int(nuke.root().knob('fps').getValue()))
    s['FrameRangeMin'].setValue(int(nuke.Root()['first_frame'].value())) 
    s['FrameRangeMax'].setValue(int(nuke.Root()['last_frame'].value())) 


    s['FPS'].setFlag(nuke.STARTLINE)
    s['FrameRangeMax'].clearFlag(nuke.STARTLINE)
    s['UpdatePath'].setFlag(nuke.STARTLINE)
    # Set BBox knobs flags and tooltips
    s['BBoxX'].setFlag(nuke.STARTLINE)
    s['BBoxY'].clearFlag(nuke.STARTLINE)
    s['BBoxW'].setFlag(nuke.STARTLINE)
    s['BBoxH'].clearFlag(nuke.STARTLINE)
    s['BBoxX'].setTooltip("Top-left X coordinate of the bounding box.")
    s['BBoxY'].setTooltip("Top-left Y coordinate of the bounding box.")
    s['BBoxW'].setTooltip("Width of the bounding box.")
    s['BBoxH'].setTooltip("Height of the bounding box.")

    s['GenerateMask'].setFlag(nuke.STARTLINE)
    
    s['FPS'].setTooltip("Target FPS for the output video")
    s['ModelType'].setTooltip("Choose your model type")
    s['OutputPath'].setTooltip("path/to/your/file_####.exr, to create an image sequence add #### or ### ")
    s['GenerateMask'].setTooltip("Generate Mask")
    