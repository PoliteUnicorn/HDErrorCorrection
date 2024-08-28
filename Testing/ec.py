'''
This class will take in a string set of reads and perform the initial round
of error correction. It will do this by 
(1) Vectorize every k-mer in the input 
(2) Store them in a dictionary 
(3) Identify the unique kmers 
(4) Replace the kmers that only appear once with their most similar kmer
(5) Write the output into a file 
'''

import csv
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class HDErrorCorrection:
    '''
    Constructor to set up values and create base hypervectors. 

    k                   = length of a k-mer
    D                   = dimension of the hypervector
    minCoverage         = the minimum coverage amount for a k-mer to not be considered an error 
    coverage_amplifier  = shortest that a k-mer can be before it is deemed unique 
    '''
    def __init__(self, k: int, D: int, min_coverage=2, coverage_amplifier=1.5):
        '''
        Check that the dimension is large enough, otherwise 
        different encoded kmer hypervectors will be too similar 
        '''
        if D < 10 * k:
            raise ValueError("Please choose a larger value for the dimension")

        '''
        Initialize variables 
        '''
        self.k = k
        self.D = D
        self.min_coverage = min_coverage
        self.coverage_amplifier = coverage_amplifier
        self.high_freq_dict = {}
        self.total_kmers = 0
        
        '''
        Dictionary to store all unique kmers + high frequency kmers
        '''
        self.unique_kmers = {}
        self.all_kmers = {}

        ''' 
        create the vectors for A, C, G, and T
        each will be dimension D 
        store the encoding scheme in a dictionary (all_bases)
        '''
        self.all_bases = {}

        self.vector_a = []
        self.vector_c = []
        self.vector_t = []
        self.vector_g = []
        for i in range(0, D):
            self.vector_a.append(random.choice([-1, 1]))
            self.vector_c.append(random.choice([-1, 1]))
            self.vector_t.append(random.choice([-1, 1]))
            self.vector_g.append(random.choice([-1, 1]))

        #set all
        self.all_bases['A'] = self.vector_a
        self.all_bases['C'] = self.vector_c
        self.all_bases['G'] = self.vector_g
        self.all_bases['T'] = self.vector_t

        self.start_time = time.time()
    

    '''
    Checks if a kmer is unique and adds it to the dictionary, 
    otherwise pops it out of the dictionary

    self.unique_kmers[kmer]: [hv, read_num]

    returns: if the kmer is unique 
    '''
    def is_unique(self, query_kmer:str, hv, read_num:int):
        
        # check if query kmer is in 
        if query_kmer in self.unique_kmers:
            self.unique_kmers.pop(query_kmer)
            return False
        else: 
            self.unique_kmers[query_kmer] = [hv, read_num]
            return True

    '''
    This is the encoding method used on the first kmer of each read 

    kmer: the kmer to encode 
    returns: the hypervectors of the encoded kmer 
    '''
    def vectorize(self, kmer: str):
        curr_kmer = np.ones(self.D)

        # if kmer is not length k
        if len(kmer.strip()) != self.k:
            raise ValueError("k-mer not of length k")

        # iterates through each base in the kmer to create hypervector 
        for i in range(len(kmer.strip())):
            if kmer[i] in {'A', 'C', 'G', 'T'}:
                curr_kmer = curr_kmer * (np.roll(self.all_bases[kmer[i]], i))
            else:
                raise ValueError("k-mers contain invalid character")

        return curr_kmer

    
    '''
    Combines all high frequency kmers from a single read and turns it into a hypervector 
    
    read: string that contains the single read 
    returns a hv that contains all of the high frequency reads summed together, and a dictionary

    self.high_frequency_hv[read_num] = [combined_hv]
    '''
    def combine_hv(self, read:str, read_num:int):
        # create temp dict for all kmers 
        temp_high_freq_dict = {}
        sum_hv = np.zeros(self.D)

        if len(read) < self.k:
            print('len read = ', len(read))
            raise ValueError(f'read must have length >= {self.k}')
        
        '''
        encode the first kmer to use for all other kmers 
        '''
        # get the first kmer and encode it 
        curr_kmer = read[:self.k]
        curr_first_base = read[0]
         
        # check if it is unique and get the hypervector 
        if curr_kmer not in self.unique_kmers and curr_kmer not in self.high_freq_dict:
            curr_hv = self.vectorize(curr_kmer)
            self.unique_kmers[curr_kmer.strip()] = [curr_hv, read_num] 
       
        # otherwise if it is in the unique kmers already, don't do it again 
        elif curr_kmer in self.unique_kmers:
            curr_hv = self.unique_kmers[curr_kmer][0]
            self.unique_kmers.pop(curr_kmer)
            # add to high freq dict if not added already 
            if curr_kmer not in self.high_freq_dict:
                self.high_freq_dict[curr_kmer.strip()] = curr_hv
                temp_high_freq_dict[curr_kmer] = curr_hv

                sum_hv += curr_hv
        
        # otherwise it must be in the high_freq_dict
        elif curr_kmer in self.high_freq_dict:
            curr_hv = self.high_freq_dict[curr_kmer]
            
        self.total_kmers += 1
        # use GenieHD rotational method 
        for i in range(1, len(read) - self.k + 1):
            # store the last kmer's data 
            prev_first_base = curr_first_base
            prev_hv = curr_hv
        
            # get current kmer 
            curr_kmer = read[i:i + self.k]
            curr_first_base = read[i]

            # if not in unique_kmers then use n bit rotation and then add it 
            if curr_kmer.strip() not in self.unique_kmers and curr_kmer not in self.high_freq_dict:
                curr_hv = prev_hv * self.all_bases[prev_first_base] # cancel out the first base
                curr_hv = np.roll(curr_hv, -1) # unrotate everything?
                curr_hv = curr_hv * np.roll(self.all_bases[curr_kmer[-1]], self.k - 1) # multiply by the last base 
                self.unique_kmers[curr_kmer.strip()] = [curr_hv, read_num] 
        
            elif  curr_kmer.strip()  in self.unique_kmers:
                # if the current kmer appears again from the unique kmers 
                curr_hv = self.unique_kmers[curr_kmer][0]
                self.unique_kmers.pop(curr_kmer)
                '''
                add to high freq dict if not added already 

                high_freq_dict[kmer] = hv
                '''
                if curr_kmer not in self.high_freq_dict:
                    self.high_freq_dict[curr_kmer] = curr_hv
                    temp_high_freq_dict[curr_kmer] = curr_hv
                    sum_hv += curr_hv
            
            self.total_kmers += 1  

        self.hf = temp_high_freq_dict
        self.sv = sum_hv
        return sum_hv, temp_high_freq_dict

    '''
    Creates a List of all hvs for each read 
    takes in a file name of all reads  
    '''
    def encode_all_reads(self, file:str):
        self.high_freq_hv_dict = [] # takes form self.high_freq_hv_dict [read_num]: {hv}, takes the hypervector of all the kmers combined 
        self.all_reads_dict = [] # takes form self.all_reads_dict [read_num]: {kmers: hvs}, takes all of the high frequency vectors and their corresponding kmers 
        all_reads = open(file, 'r')
        read_num = 0

        self.start_time = time.time()
        
        # call the combine_hv for each read 
        for read in all_reads.readlines():
            read = read.strip()  # Ensure to strip out extra whitespace and newlines
            hv, read_dict = self.combine_hv(read, read_num)
            self.high_freq_hv_dict.append(hv)
            self.all_reads_dict.append(read_dict)
            read_num += 1
        
        self.end_time = time.time()

        # print(self.high_freq_hv_dict)
        print('k=', self.k, 'total time taken: ', time.time() - self.start_time)
        
        return self.end_time - self.start_time 

    '''
    find most similar kmer by querying with each of the reads 
    if it passes the read threshold then add to a list for largest
    otherwise, keep going 

    takes in a unique kmer's hypervector 
    returns: a list containing the positions of all of the all of the most similar possible reads 
    # '''
    def find_most_similar_reads(self, unique_kmer_hv):
        highest_dot_product = 0
        similar_reads = []
        
        for idx, hv in enumerate(self.high_freq_hv_dict):  # check if they are similar enough 
            dot_product = abs(np.dot(unique_kmer_hv, hv))
            
            # if dot_product > self.D * 0.5 and dot_product >= highest_dot_product:
            if dot_product > self.D * 0.5 and dot_product >= highest_dot_product:
                similar_reads.append(idx)
                highest_dot_product = dot_product
        
        return similar_reads
    
    '''
    find most similar kmer given a hypervector a list of similar reads 
    '''
    def find_most_similar_kmer(self, unique_kmer_hv, similar_reads:list):
        most_similar_kmer = ''
        highest_dp = 0
        # go through each read in the list given its number '
        for read_num in similar_reads:
            '''
            self.all_reads_dict [read_num]: {kmers: hvs}
            '''
            for kmer in self.all_reads_dict[read_num]:
                # print(kmer)
                if np.dot(self.all_reads_dict[read_num][kmer], unique_kmer_hv) > highest_dp:
                    most_similar_kmer = kmer

        return most_similar_kmer

    # '''
    # Go through all of the unique_kmers and replace them using the find_most_similar_kmer function
    # '''
    # def replace_all_unique_kmers(self, write_file:str):
       
    #     for unique in self.unique_kmers:
    #        reads_to_search = self.find_most_similar_reads(unique)
    #        replacement_kmer = self.find_most_similar_kmer(unique, reads_to_search)
    #        # go into the read csv file and replace it 
           
    def replace_all_unique_kmers(self, read_file:str, write_file: str):
        start = time.time()
        # Create a temporary list to store updated reads
        updated_reads = []

        # Open the input file containing the reads
        with open(read_file, 'r') as file:
            reads = file.readlines()

        # Process each read
        for read in reads:
            read = read.strip()  # Remove any extra whitespace or newline characters
            updated_read = read
            
            # Replace all unique kmers in the read
            for unique_kmer in self.unique_kmers:
                # Find the most similar kmer to replace the unique kmer
                similar_kmer = self.find_most_similar_kmer(self.unique_kmers[unique_kmer][0], 
                                                        self.find_most_similar_reads(self.unique_kmers[unique_kmer][0]))
                
                if similar_kmer != '':
                    # Replace all occurrences of the unique kmer with the most similar kmer
                    updated_read = updated_read.replace(unique_kmer, similar_kmer)
            
            # Add the updated read to the list
            updated_reads.append(updated_read)

        # Write the updated reads to a new file
        with open(write_file, 'w') as file:
            for updated_read in updated_reads:
                file.write(updated_read + '\n')

        return time.time() - start

    
    def compare_csv_files(self, file1: str, file2: str):
        """Compare two CSV files line by line to see the exact number of differing characters."""
        total_diff = 0

        # Open both CSV files
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            reader1 = csv.reader(f1)
            reader2 = csv.reader(f2)
            
            # Compare each line
            for line1, line2 in zip(reader1, reader2):
                # Join the elements of each line to compare them as single strings
                line1_str = ''.join(line1)
                line2_str = ''.join(line2)

                # Compare the lines character by character
                max_length = max(len(line1_str), len(line2_str))
                line1_str = line1_str.ljust(max_length)
                line2_str = line2_str.ljust(max_length)

                # Count differing characters
                diff_count = sum(c1 != c2 for c1, c2 in zip(line1_str, line2_str))
                total_diff += diff_count

        return total_diff





k = 15
D = 10000
he = HDErrorCorrection(k, D)
he.encode_all_reads('reads-with-errors.csv')
he.replace_all_unique_kmers('reads-with-errors.csv', 'corrected-reads.csv')

