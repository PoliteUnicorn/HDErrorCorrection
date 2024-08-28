import csv
import random
import time
import numpy as np
import matplotlib.pyplot as plt

class HDErrorCorrection:
    def __init__(self, k: int, D: int, min_coverage=2, coverage_amplifier=1.5):
        if D < 10 * k:
            raise ValueError("Please choose a larger value for the dimension")

        self.k = k
        self.D = D
        self.min_coverage = min_coverage
        self.coverage_amplifier = coverage_amplifier
        self.high_freq_dict = {}
        self.total_kmers = 0
        
        self.unique_kmers = {}
        self.all_kmers = {}

        self.all_bases = {}
        self.vector_a = []
        self.vector_c = []
        self.vector_t = []
        self.vector_g = []
        for i in range(D):
            self.vector_a.append(random.choice([-1, 1]))
            self.vector_c.append(random.choice([-1, 1]))
            self.vector_t.append(random.choice([-1, 1]))
            self.vector_g.append(random.choice([-1, 1]))

        self.all_bases['A'] = self.vector_a
        self.all_bases['C'] = self.vector_c
        self.all_bases['G'] = self.vector_g
        self.all_bases['T'] = self.vector_t

        self.start_time = time.time()

    def vectorize(self, kmer: str):
        curr_kmer = np.ones(self.D)
        if len(kmer.strip()) != self.k:
            raise ValueError("k-mer not of length k")
        for i in range(len(kmer.strip())):
            if kmer[i] in {'A', 'C', 'G', 'T'}:
                curr_kmer = curr_kmer * (np.roll(self.all_bases[kmer[i]], i))
            else:
                raise ValueError("k-mers contain invalid character")
        return curr_kmer

    def combine_hv(self, read: str, read_num: int):
        temp_high_freq_dict = {}
        sum_hv = np.zeros(self.D)
        if len(read) < self.k:
            raise ValueError(f'read must have length >= {self.k}')
        curr_kmer = read[:self.k]
        curr_first_base = read[0]
        if curr_kmer not in self.unique_kmers and curr_kmer not in self.high_freq_dict:
            curr_hv = self.vectorize(curr_kmer)
            self.unique_kmers[curr_kmer.strip()] = [curr_hv, read_num]
        elif curr_kmer in self.unique_kmers:
            curr_hv = self.unique_kmers[curr_kmer][0]
            self.unique_kmers.pop(curr_kmer)
            if curr_kmer not in self.high_freq_dict:
                self.high_freq_dict[curr_kmer.strip()] = curr_hv
                temp_high_freq_dict[curr_kmer] = curr_hv
                sum_hv += curr_hv
        elif curr_kmer in self.high_freq_dict:
            curr_hv = self.high_freq_dict[curr_kmer]
        self.total_kmers += 1
        for i in range(1, len(read) - self.k + 1):
            prev_first_base = curr_first_base
            prev_hv = curr_hv
            curr_kmer = read[i:i + self.k]
            curr_first_base = read[i]
            if curr_kmer.strip() not in self.unique_kmers and curr_kmer not in self.high_freq_dict:
                curr_hv = prev_hv * self.all_bases[prev_first_base]
                curr_hv = np.roll(curr_hv, -1)
                curr_hv = curr_hv * np.roll(self.all_bases[curr_kmer[-1]], self.k - 1)
                self.unique_kmers[curr_kmer.strip()] = [curr_hv, read_num]
            elif curr_kmer.strip() in self.unique_kmers:
                curr_hv = self.unique_kmers[curr_kmer][0]
                self.unique_kmers.pop(curr_kmer)
                if curr_kmer not in self.high_freq_dict:
                    self.high_freq_dict[curr_kmer] = curr_hv
                    temp_high_freq_dict[curr_kmer] = curr_hv
                    sum_hv += curr_hv
            self.total_kmers += 1
        self.hf = temp_high_freq_dict
        self.sv = sum_hv
        return sum_hv, temp_high_freq_dict

    def encode_all_reads(self, file: str):
        self.high_freq_hv_dict = []
        self.all_reads_dict = []
        with open(file, 'r') as all_reads:
            read_num = 0
            self.start_time = time.time()
            for read in all_reads.readlines():
                read = read.strip()
                hv, read_dict = self.combine_hv(read, read_num)
                self.high_freq_hv_dict.append(hv)
                self.all_reads_dict.append(read_dict)
                read_num += 1
            self.end_time = time.time()
        return self.end_time - self.start_time

    def replace_all_unique_kmers(self, read_file: str, write_file: str):
        start = time.time()
        updated_reads = []
        with open(read_file, 'r') as file:
            reads = file.readlines()
        for read in reads:
            read = read.strip()
            updated_read = read
            for unique_kmer in self.unique_kmers:
                similar_kmer = self.find_most_similar_kmer(self.unique_kmers[unique_kmer][0],
                                                        self.find_most_similar_reads(self.unique_kmers[unique_kmer][0]))
                if similar_kmer != '':
                    updated_read = updated_read.replace(unique_kmer, similar_kmer)
            updated_reads.append(updated_read)
        with open(write_file, 'w') as file:
            for updated_read in updated_reads:
                file.write(updated_read + '\n')
        return time.time() - start

    def find_most_similar_reads(self, unique_kmer_hv):
        highest_dot_product = 0
        similar_reads = []
        for idx, hv in enumerate(self.high_freq_hv_dict):
            dot_product = abs(np.dot(unique_kmer_hv, hv))
            if dot_product > self.D * 0.5 and dot_product >= highest_dot_product:
                similar_reads.append(idx)
                highest_dot_product = dot_product
        return similar_reads

    def find_most_similar_kmer(self, unique_kmer_hv, similar_reads: list):
        most_similar_kmer = ''
        highest_dp = 0
        for read_num in similar_reads:
            for kmer in self.all_reads_dict[read_num]:
                if np.dot(self.all_reads_dict[read_num][kmer], unique_kmer_hv) > highest_dp:
                    most_similar_kmer = kmer
        return most_similar_kmer

    def measure_performance(self, k, num_reads, base_file):
        self.k = k  # Ensure k is set to the fixed length
        self.total_kmers = 0  # Reset total kmers count

        # Create a temporary file with the desired number of reads
        temp_file = f"temp_reads_{num_reads}.txt"
        self.create_temp_file(base_file, num_reads, temp_file)

        # Measure encoding time
        encoding_time = self.encode_all_reads(temp_file)
        kmers_count = self.total_kmers

        # Measure replacement time
        replacement_time = self.replace_all_unique_kmers(temp_file, 'corrected-reads.csv')

        return kmers_count, encoding_time, replacement_time

    def create_temp_file(self, base_file, num_reads, temp_file):
        with open(base_file, 'r') as f:
            all_lines = f.readlines()

        if num_reads > len(all_lines):
            num_reads = len(all_lines)  # Adjust num_reads to the maximum available lines

        # Sample the required number of lines from the base file
        sampled_lines = random.sample(all_lines, num_reads)

        with open(temp_file, 'w') as f:
            f.writelines(sampled_lines)

# Define k values and number of reads to test
k = 100
num_reads_list = [100, 200, 300]
base_file = 'reads-with-errors.csv'

# Initialize the HDErrorCorrection instance
he = HDErrorCorrection(k=k, D=10000)  # Use a fixed k for consistent length

# Measure performance for different numbers of reads
kmers_counts = []
encoding_times = []
replacement_times = []

for num_reads in num_reads_list:
    kmers_count, encoding_time, replacement_time = he.measure_performance(k, num_reads, base_file)
    kmers_counts.append(kmers_count)
    encoding_times.append(encoding_time)
    replacement_times.append(replacement_time)

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(num_reads_list, encoding_times, marker='o', linestyle='-', color='b', label='Encoding Time')
plt.xlabel('Number of Reads')
plt.ylabel('Time (seconds)')
plt.title(f'Encoding Time vs Number of Reads (k={k})')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(num_reads_list, replacement_times, marker='o', linestyle='-', color='r', label='Replacement Time')
plt.xlabel('Number of Reads')
plt.ylabel('Time (seconds)')
plt.title(f'Replacement Time vs Number of Reads (k={k})')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
