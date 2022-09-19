import unittest
import neuron_igraph_analysis as nrn_graph

from neuprint import Client
from neuprint import fetch_neurons, NeuronCriteria as NC, merge_neuron_properties
from neuprint import fetch_skeleton, fetch_synapses, fetch_synapse_connections, attach_synapses_to_skeleton, skeleton
from neuprint.utils import connection_table_to_matrix


class TestNeuronIgraphAnalysis(unittest.TestCase):

    def test_neuprint_to_um(self):
        self.assertEqual(nrn_graph.neuprint_to_um(125), 1, "Should be 1")


def test_vertexDist_brute_force():
    """Create a file to write the rows currently tested, then remove the row if it passes."""

    skeleton_w_syns = nrn_graph.initialize_dataframe()
    g = nrn_graph.createGraph(skeleton_w_syns)

    logfile = r'testing logs\logfile.txt'
    outputsyn = 13018

    with open(logfile, 'r+') as file:
        # get rowId to start testing from
        lines = file.readlines()
        file.seek(0)
        if len(lines) == 0: # No failed rows yet
            startRowId = int(skeleton_w_syns.loc[0, 'rowId'])
        else:
            last_line = file.readlines()[-1]
            startRowId = int(last_line.strip()) + 1
        
        # test each row, first write rowId to the file, then test, on the next loop remove prevRow if possible
        count = 0
        prevRow = None
        for index, row in skeleton_w_syns.loc[skeleton_w_syns['rowId'] >= startRowId].iterrows():
            # Remove previous row from file
            if prevRow is not None:
                file.seek(0) # Set file pointer back to start
                for line in lines:
                    if int(line.strip()) != prevRow:
                        file.write(line + '\n')
                    else:
                        print('deleted prev row')
            
            # Add current row to file
            file.write(str(row['rowId']) + '\n')
            prevRow = int(row['rowId'])
            
            # Test if current row fails
            if row['structure'] == 'post':
                skeleton_w_syns.loc[index, 'distance'] = nrn_graph.vertexDist(g, g.vs[row['rowId']], g.vs[outputsyn])


if __name__ == '__main__':
    unittest.main()