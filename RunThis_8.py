#!/usr/local/bin/python3

# https://realpython.com/python-gui-tkinter/#building-a-text-editor-example-app
import harvestExpLearn8a as harvest
import tkinter as tk
import tkinter.ttk as ttk
import time
import sys
from tkinter.filedialog import askopenfilename, asksaveasfilename
import os
from os import path
import pickle
import csv
from multiprocessing import Process, freeze_support


########## from https://www.edureka.co/community/43417/stdout-to-tkinter-gui
class StdoutRedirector(object):
    def __init__(self,main_large_text_field):
        self.main_large_text_field = main_large_text_field

    def write(self,string):
        self.main_large_text_field.insert(tk.END, string)
        self.main_large_text_field.see(tk.END)

###############################
		
class RunThis:
	
	def __init__(self, master=None):
#		 Frame.__init__(self, master)
		self.window = master
		self.pickle_db_name = "pickle_db_8"
		self.script_path = ''
#		  self.trainingSets = ''
		self.target_file_path = ''
		self.stopwords_file_path = ''
		self.trainingSets=	[]
		self.training_file_path0 = ''
		self.training_file_path1 = ''
		self.training_file_path2 = ''
		self.script_key = 'script'
		self.target_key = 'target'
		self.stopwords_key = 'stopwords'
		self.training_file_key = 'training'
		self.training_file_key0 = 'training0'
		self.training_file_key1 = 'training1'
		self.training_file_key2 = 'training2'
		self.buttons_frame = tk.Frame(window, relief=tk.RAISED, bd=2)
		
		self.script_path_label = self.create_label(1, 'blue', self.script_path) 
		self.target_path_label = self.create_label(1,'green', self.target_file_path)
		self.stopwords_file_path_label = self.create_label(1, 'blue', self.stopwords_file_path)
		self.training_sets_path_label = self.create_label(1, 'green', self.training_file_path0)
		
		self.main_large_text_field = tk.Text(window, fg = "blue")
		self.gui_list = []
#		  self.update_cwd()

#		  self.retrieve_file_paths_from_database()

		#,global main_large_text_field
		self.style = ttk.Style()
		self.style.configure('B.TButton', foreground='blue')
		self.style.configure('G.TButton', foreground='green')
		self.style.configure('R.TButton', foreground='red')         
		
		os.fdopen(sys.stdin.fileno(), 'rb', buffering=0)
		sys.stderr = sys.stdout
		sys.stdout = StdoutRedirector(self.main_large_text_field)

		self.go()
	
	def go(self):
		self.create_window()
		self.add_paths_and_buttons()
		self.build_menus()
		self.setup_database()
		
#		 self.print_file_paths()
#		 self.display_file_paths()

	def __repr__(self):
		return (f'{self.__class__.__name__}')

	   
	def create_label(self, line_scale, color, text):
#		  ltgray = '#f2f2f2'
		number_of_lines = 300
		label = tk.Label(self.buttons_frame, text='No file(s) currently selected.\nClick the button above to select.',
						 fg=color, wraplength = number_of_lines*line_scale, justify=tk.LEFT)
#		  label.configure(state=tk.NORMAL)	  
		return label

#	  def create_text_field(self, line_scale, color, text):
#		  ltgray = '#f2f2f2'
#		  number_of_lines = 4
#		  text_obj = tk.Text(self.buttons_frame, height=number_of_lines*line_scale, width=23, fg=color, bg = ltgray)
#		  if text =='':
#			  text_obj.insert(tk.END, 'No file(s) currently selected.\nClick the button above to select.')
#		  text_obj.configure(state=tk.NORMAL)	 
#		  return text_obj


	def setup_database(self):
#		   https://docs.python.org/3/library/dbm.html#module-dbm
		print("setup_database")
		self.script_key = 'script'
		self.target_key = 'target'
		self.stopwords_key = 'stopwords'
		self.training_key = 'training'
#		  os.chdir(os.path.dirname(__file__))
		print(os.getcwd())
		db_file = os.getcwd()+"/"+self.pickle_db_name
		#db_file = os.getcwd()+"/toadstool"
		if path.exists(db_file):
#			 print("RT 87 db_path :", db_file)
			size = os.path.getsize(db_file)
			if (size > 0):
				self.get_file_paths_from_database()	   

		'''''		 
		if path.exists(db_path):
			self.db_has_data = True
#			  self.get_file_paths_from_database()
			self.get_file_paths_from_database			 
		else:			 
			self.db_has_data = False
#			  self.pickle_db_name = open(self.pickle_db_name, 'ab')
			self.get_file_paths_from_database			 
			self.db_has_data = True
		'''
#  
	def store_file_paths_in_database(self):
		try:
			self.pickle_write_db_name = open(self.pickle_db_name, 'wb')
#			  db = {}
#			  db[self.script_key] = self.script_path
#			  db[self.target_key] = self.target_file_path
#			  db[self.stopwords_key] = self.stopwords_file_path o
			
			pickle.dump(self.script_path, self.pickle_write_db_name)
			pickle.dump(self.target_file_path, self.pickle_write_db_name)
			pickle.dump(self.stopwords_file_path, self.pickle_write_db_name)
#			 self.number_training_files_as_string = str( len(self.trainingSets))
			number_training_files = len(self.trainingSets)
			number_training_files_as_str = str(number_training_files)
			print("RT 151 number of training files: ", number_training_files_as_str)
#			 pickle.dump(number_training_files_as_str,	self.pickle_db_name)
			pickle.dump(self.trainingSets, self.pickle_write_db_name)
			#for training_path in self.trainingSets:
			#	  pickle.dump(training_path, self.pickle_write_db_name)
			#	  print("RT 153 training_path", training_path)
#			  pickle.dump(str(self.trainingSets), self.pickle_db_name)		  
			self.pickle_write_db_name.close()
		except EOFError as eof_err:
			print("RT 154 EOF error: {0}".format(eof_err))
		except OSError as err:
			print("RT 156 OS error: {0}".format(err))
		except ValueError:
			print("RT 158 Could not convert data to an integer.")
		except TypeError:
			print("RT 160 type error:", sys.exc_info()[0])
		except pickle.PickleError:
			print("RT 162 pickle error:", sys.exc_info()[0])
		
	def get_file_paths_from_database(self):
		try:
			self.pickle_open_db_name = open(self.pickle_db_name, 'rb')
#			  db = pickle.load(self.pickle_db_name)
#			  self.script_path = db[self.script_key]
#			  self.target_file_path = db[self.target_key]
#			  self.stopwords_file_path = db[self.stopwords_key]
			self.script_path = pickle.load(self.pickle_open_db_name)
			self.target_file_path = pickle.load(self.pickle_open_db_name)
			self.stopwords_file_path = pickle.load(self.pickle_open_db_name)
#			 number_training_files	= pickle.load(self.pickle_db_name)
#			 number_training_files_as_string  = pickle.load(self.pickle_db_name)
#			 print("RT 181 number of elements: ", number_training_files_as_string)
#			 number_training_files = int(number_training_files_as_string)
#			  number_training_files = 10
			self.trainingSets = pickle.load(self.pickle_open_db_name)
			#for path in self.trainingSets(range = number_training_files):
			#for path in self.trainingSets:
			for path1 in self.trainingSets:
				print(path1)
				 #pickle.dump(self.training_file_path+Str(index), self.pickle_db_name)

			self.pickle_open_db_name.close()
		except TypeError as err:
			print("RT 188 TypeError: ", err)
		except pickle.UnpicklingError:
			print("RT 150 pickle error:", sys.exc_info()[0])
		self.display_file_paths()
 
#			  self.script_path = db[self.script_key]
#			  self.target_file_path = db[self.target_key]
#			  self.stopwords_file_path = db[self.stopwords_key]
#			  try:
#				  a = 7
# #					self.trainingSets = db[self.training_key]
# #					print("159 self.trainingSets_x: ", self.trainingSets_x)
# #					self.trainingSets = ast.literal_eval(self.trainingSets_x)
#			  except TypeError:
#				  print("RT 171 type error:", sys.exc_info()[0])

#			  self.pickle_db_name.close()
#		  except EOFError as eof_err:
#			  print("RT 165 EOF error: {0}".format(eof_err))
#		  except OSError as err:
#			  print("RT 167 OS error: {0}".format(err))
#		  except ValueError:
#			  print("RT 169 OS Could not convert data to an integer.")
			 
	def display_file_paths(self):
		print('Entering display_file_paths()')
		self.print_file_paths()
		self.script_path_label.configure(text = self.script_path)
		self.target_path_label.configure(text = self.target_file_path)		 
		self.stopwords_file_path_label.configure(text = self.stopwords_file_path)
		files = ''
		for path1 in self.trainingSets:
			file = os.path.basename(path1)+'\n'
			files += file
		self.training_sets_path_label.configure(text = files, wraplength = 300)
#		  self.training_sets_path_label.configure(text = str(self.trainingSets), wraplength = 1200)

	def set_individual_training_files(self):
		index = 0
		print("In Set invidual training files", self.trainingSets)
		for file in self.trainingSets:
			if index == 0:
				self.training_file_path0 = file
			if index == 1:
				self.training_file_path1 = file
			if index == 2:
				self.training_file_path2 = file
			index +=1
			
	def print_file_paths(self):
#		  self.get_file_paths_from_database()
#		  self.get_individual_training_files()
		print("\n")
#		  print(self.cwd: \n", self.cwd, "\n")
		print("self.script_path: \n", self.script_path, "\n")
		print(" self.target_file_path: \n", self.target_file_path, "\n")
		print(" self.stopwords_file_path: \n", self.stopwords_file_path, "\n")
		for path1 in self.trainingSets:
			print(path1)

	def select_a_file(self, file_type_string):
		"""Select calculation script file."""
		selected_file_path = askopenfilename(
			filetypes=[("A file", file_type_string)]
		)
		if not selected_file_path:
			return
		with open(selected_file_path, "r") as input_file:
			text = input_file.read()
			self.main_large_text_field.delete(1.0, tk.END)
			self.main_large_text_field.insert(tk.END, text)
		window.title(f"{selected_file_path}")
		print("\n\nRT 206 selected_file_path:",selected_file_path)
		return selected_file_path
  
	def select_python_script(self):
		"""Select calculation script file."""
		self.script_path = self.select_a_file("*.py")
		self.display_file_paths()
		self.db_has_data= True

	def select_target_file(self):
		"""Select a target file."""
		self.target_file_path = self.select_a_file("*.csv")
		self.display_file_paths()
		self.db_has_data= True
		
	def select_stopwords_file(self):
		"""Open stopwords file for reading."""
		self.stopwords_file_path = self.select_a_file("*.txt")
		self.display_file_paths()
		self.db_has_data= True

	def convert_tuple_into_list(self,tpl):
		list = []
		for element in tpl:
			list.append(element)
		return list

	def select_training_files(self):
		"""Select training files."""
		training_file_paths = askopenfilename(multiple=True,
			filetypes=[("CSV Files", "*.csv")]
		)
		if not training_file_paths:
			return
		self.trainingSets = training_file_paths
		self.main_large_text_field.delete(1.0, tk.END)
		print(training_file_paths)
		trainingTuples = training_file_paths
		trainingList = self.convert_tuple_into_list(trainingTuples)
		print(trainingList)
		self.trainingSets = trainingList # actually, a list
		print("RT 292 self.trainingSets: ", self.trainingSets)
#		  self.training_file_paths = self.select_a_file("*.py")
#		  for file_path in trainingList:
#			  self.main_large_text_field.insert(tk.END, file_path +"\n")
		self.set_individual_training_files()
		self.display_file_paths()
		window.title(f"trainingFiles")
		


	def run_harvest(self):
		global argv
	
		""" run the calculation """
		window.title(f"Calculations")
		sys.argv = [self.script_path, self.trainingSets, "-t", self.target_file_path, "-s", self.stopwords_file_path]
		print("RunThis 315 self.stopwords_file_path	 :", self.stopwords_file_path)
#		  text = self.main_large_text_field.get(1.0, tk.END)
		self.main_large_text_field.replace(1.0, tk.END, "Starting the calculation...")
		print("RT 323 Starting the calculation.")
		timeStart = time.time()
		window.title(f"Calculating")
#		  from multiprocessing import Process
#		  p = Process(target=harvest.run_harvest_orig, args=())
#		  p.start()
#		  p.join()
		
		harvest.run_harvest_orig()
		timeEnd = time.time()
		elapsedTime = int(timeEnd-timeStart)
		result = "Finished the calculation in " + str(elapsedTime) + " seconds"
#		self.main_large_text_field.insert(tk.END, result)
#		print("RT 336 ", result)
#
#		 self.main_large_text_field.insert(tk.END, result)
		
	def open_text_file(self):
		"""Open a file for editing."""
		filepath = askopenfilename(
				filetypes=[("Text Files", "*.txt")]
			)
		if not filepath:
			return
		self.main_large_text_field.delete(1.0, tk.END)
		with open(filepath, "r") as input_file:
			text = input_file.read()
			self.main_large_text_field.insert(tk.END, text)
			window.title(f"{filepath}")


	def open_result_file(self):
		"""Open a file."""
		filepath = askopenfilename(
			filetypes=[("CSV Files", "*.csv")]
		)
		if not filepath:
			return
#		displayCSVFile(filepath)
		self.main_large_text_field.delete(1.0, tk.END)
		with open(filepath, "r") as output_result_file:
			text = output_result_file.read()
			self.main_large_text_field.insert(tk.END, text)
		window.title(f"Result - {filepath}")
		
	def save_file(self):
		"""Save the current file as a new file."""
		filepath = asksaveasfilename(
			defaultextension="csv",
			filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
		)
		if not filepath:
			return
		with open(filepath, "w") as output_file:
			text = self.main_large_text_field.get(1.0, tk.END)
			output_file.write(text)
		window.title(f"Simple Text Editor - {filepath}")
		
	def quit(self):
		""" quit the program """
#		  self.pickle_db_name.close()
#		  self.db.close()
		self.store_file_paths_in_database()
		sys.stdout = sys.__stdout__
		window.destroy()

	def create_window(self):
		window.title("Run This => BuildABear")
		window.rowconfigure(0, minsize=600, weight=1)
#		window.rowconfigure(1, minsize=400, weight=1)
		window.columnconfigure(1, minsize=1200, weight=1)


	def build_menus(self):
		menubar = tk.Menu(window)
		filemenu = tk.Menu(menubar, tearoff=0)
		filemenu.add_command(label="Select Python Script...", command=self.select_python_script, accelerator="Command+1")
		filemenu.add_command(label="Select Target CSV File...", command=self.select_target_file, accelerator="Command-2")
		filemenu.add_command(label="Select Stopwords Text File...", command=self.select_stopwords_file, accelerator="Command-3")
		filemenu.add_command(label="Select Training CSV Files...", command=self.select_training_files, accelerator="Command-4")
	#	  window.bind('<Command-3>', select_training_files)
		filemenu.add_separator()
		filemenu.add_command(label="Run", command=self.run_harvest, accelerator="Command+5")
		filemenu.add_separator()
		filemenu.add_command(label="display_file_paths", command=self.display_file_paths, accelerator="Command+")
		filemenu.add_command(label="store_file_paths_in_database", command=self.store_file_paths_in_database, accelerator="Command+")
		filemenu.add_command(label="get_file_paths_from_database", command=self.get_file_paths_from_database, accelerator="Command+0")
		menubar.add_cascade(label="File", menu=filemenu)
		helpmenu = tk.Menu(menubar, tearoff=0)
		helpmenu.add_command(label="About...")
		menubar.add_cascade(label="Help", menu=helpmenu)
		# root.option_add('*tearOff', FALSE)
		window.config(menu=menubar)


	def create_button(self, txt, cmd, a_style):
		button = ttk.Button(self.buttons_frame, command=cmd, text=txt, style=a_style)
		return button

	def add_paths_and_buttons(self):
		# sys.argv = [self.script_path, self.trainingSets, "-t", self.target_file_path, "-s", self.stopwords_file_path]
		button_load_script = self.create_button("1. Select Python Script (e.g. harvest...)...", self.select_python_script, 'B.TButton') 
		button_open_target = self.create_button("2. Select Target...", self.select_target_file, 'G.TButton')
		button_open_stopwords = self.create_button("3. Select Stopwords File...", self.select_stopwords_file, 'B.TButton')
		button_select_training_files = self.create_button("4. Select TrainingSets...", self.select_training_files, 'G.TButton')
		button_run = self.create_button("5. Run the calculation", self.run_harvest, 'R.TButton')
		button_open_text_file = self.create_button("Display any Text File...", self.open_text_file, 'bk.TButton')
		button_open_result = self.create_button("Open Result File...", self.open_result_file, 'bk.TButton')
		button_quit = self.create_button("Quit", self.quit, 'R.TButton')
		
		self.gui_list = [button_load_script, self.script_path_label,
						 button_open_target, self.target_path_label,
						 button_open_stopwords,self.stopwords_file_path_label,	
						 button_select_training_files, self.training_sets_path_label,
						 button_run, button_open_text_file, button_open_result, button_quit]
		self.display_gui_list()
		
	def display_gui_list(self):
		index = 0
		for object in self.gui_list:
			object.grid(row=index, column=0, sticky="EW", padx=5, pady=10)
			index = index+1
#		  print("RT 519 self.gui_list:",self.gui_list)



####### Execution starts here ########
if __name__ == '__main__':
	freeze_support()
	window = tk.Tk()
	# window.after(0)
	# dbm_filename = 'dbm_file'	  # actually dbm_file.db
	rt = RunThis(window)
	rt.buttons_frame.grid(row=0, column=0, sticky="ns")
	rt.main_large_text_field.grid(row=0, column=1, sticky="nsew")
	'''
	root = tk.Tk()
	termf = tk.Frame(root, height=400, width=500)
	termf.pack(expand=True)
#	termf.pack(fill=BOTH, expand=YES)
	wid = termf.winfo_id()
	os.system('xterm -into %d -geometry 400x200 -sb &' % wid)
	root.mainloop()
	print("__main__!")
	'''
	'''
	table_width = 500
	table_height = 400
	screen_width = root.winfo_screenwidth()
	screen_height = root.winfo_screenheight()
	x = (screen_width/2) - (table_width/2)
	y = (screen_height/2) - (table_height/2)
	window.geometry("%dx%d+%d+%d" % (width, height, x, y))
	rt.table.grid(row=0, column =2, sticky='ns')
	# layout grid objects
	'''
	window.mainloop()
