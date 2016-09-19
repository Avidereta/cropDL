import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import json
import os
import time
import datetime
import subprocess

def execute_notebooks(eval_names,
                      run_path,
                      save_path,
                      eval_names_out=None,
                      n_hidden=None,
                      filter_size=None,
                      to_html=True,
                      ):
    """
    Executes ipython notebook
    :param eval_names: list of strings, ipynb names to execute
    :param run_path: path, where all eval_names are executed
    :param save_path: path, where all eval_names are saved
    :param to_html: bool, whether to save html version of file or not
    :param eval_names_out: list of strings, names gor saving after execution. If None, then the files
     will be saved with the names in eval_names
    :param n_hidden: list of ints of length len(eval_names), parameter or notebook execution
    :param filter_size: list of elements such [X,X] of length len(eval_names), where X is a filter size,
    parameter or notebook execution
    :return:
    """
    ep = ExecutePreprocessor(timeout=-1)

    logs_path = './logs.txt'


    if eval_names_out is None:
        eval_names_out = [save_path + eval_name for eval_name in eval_names]

    for i, (eval_name, eval_name_out) in enumerate(zip(eval_names, eval_names_out)):

        cur_time = datetime.datetime.now()
        
        with open(logs_path, "a+") as logs:
            logs.write("Current eval {}, time:{} \n".format(eval_name_out, cur_time.strftime("%Y-%m-%d %H:%M")))

        # put external parameter to the executed notebook
        if n_hidden is not None:
            with open('n_hidden.txt', 'w') as outfile:
                json.dump(n_hidden[i], outfile)
            print "n_hidden = {} was used".format(n_hidden[i])

        if filter_size is not None:
            print filter_size
            with open('filter_size.txt', 'w') as outfile:
                json.dump(filter_size[i], outfile)
            print "filter_size = {} was used".format(filter_size[i][0])

        with open('eval_name.txt', 'w') as outfile:
            json.dump(eval_name_out[i], outfile)

        print "the file will be saved as {}".format(eval_name_out)

        with open(eval_name) as f:
            nb = nbformat.read(f, as_version=4)

        t_start = time.time()
        try:
            out = ep.preprocess(nb, {'metadata': {'path': run_path}})
        except CellExecutionError:
            msg = 'Error executing the notebook "%s".\n\n' % eval_name
            msg += 'See notebook "%s" for the traceback.' % eval_name_out
            print(msg)
            raise

        finally:
            #write to ipynb
            with open(eval_name_out, mode='wb') as f:
                nbformat.write(nb, f)
                print "DONE:", eval_name_out
                with open(logs_path, "a+") as logs:

                    cur_time = datetime.datetime.now().time()
                    t_execution = round((time.time() - t_start)/60.0,1)
                    logs.write("Time of execution: {} minutes\nFile saved {}, time:{} \n\n"\
                               .format(t_execution, eval_name_out, cur_time.strftime("%Y-%m-%d %H:%M")))

            # save html variant
            if to_html:
                print "Converting to html..."
                subprocess.call("jupyter nbconvert {} --to html".format(eval_name_out))
                print 'Done!'


            filelist = ['n_hidden.txt', 'filter_size.txt', 'eval_name.txt']
            for f in filelist:
                try:
                    os.remove(f)
                except:
                    print "Problem with removing the file:", f

