This is a program that can merge multiple RVC models together

Much of this code was taken from [W-Okada's Voice Changer](https://github.com/w-okada/voice-changer) 

Usage: 
1. Install requirements with 'pip install -r requirements.txt' 
2. Run the program with 'python main.py'
3. Add as many rows as desired to merge by using the 'Add Merge Slot' button, and use the 'Browse' button to select the models to merge
   * **NOTE**: They must be of the same RVC version (v1, v2, etc.) and the same sample rate (all 32k, 40k, 48k)
4. Use the slider to determine what the strength of that model in the merge will be from 1 to 100
5. Click 'Merge Models' to merge them
6. If successful, the program will pop up with the merged model name which will be saved in the base directory
