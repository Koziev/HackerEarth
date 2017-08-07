using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.Security.Cryptography;


// Copyright (c) Damien Guard.  All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. 
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// Originally published at http://damieng.com/blog/2007/11/19/calculating-crc-64-in-c-and-net
namespace DamienG.Security.Cryptography
{
    /// <summary>
    /// Implements a 64-bit CRC hash algorithm for a given polynomial.
    /// </summary>
    /// <remarks>
    /// For ISO 3309 compliant 64-bit CRC's use Crc64Iso.
    /// </remarks>
    public class Crc64 : HashAlgorithm
    {
        public const UInt64 DefaultSeed = 0x0;

        readonly UInt64[] table;

        readonly UInt64 seed;
        UInt64 hash;

        public Crc64(UInt64 polynomial = Crc64Iso.Iso3309Polynomial)
            : this(polynomial, DefaultSeed)
        {
        }

        public Crc64(UInt64 polynomial, UInt64 seed)
        {
            table = InitializeTable(polynomial);
            this.seed = hash = seed;
        }

        public override void Initialize()
        {
            hash = seed;
        }

        protected override void HashCore(byte[] array, int ibStart, int cbSize)
        {
            hash = CalculateHash(hash, table, array, ibStart, cbSize);
        }

        protected override byte[] HashFinal()
        {
            var hashBuffer = UInt64ToBigEndianBytes(hash);
            HashValue = hashBuffer;
            return hashBuffer;
        }

        public override int HashSize { get { return 64; } }

        protected static UInt64 CalculateHash(UInt64 seed, UInt64[] table, IList<byte> buffer, int start, int size)
        {
            var hash = seed;
            for (var i = start; i < start + size; i++)
                unchecked
                {
                    hash = (hash >> 8) ^ table[(buffer[i] ^ hash) & 0xff];
                }
            return hash;
        }

        static byte[] UInt64ToBigEndianBytes(UInt64 value)
        {
            var result = BitConverter.GetBytes(value);

            if (BitConverter.IsLittleEndian)
                Array.Reverse(result);

            return result;
        }

        static UInt64[] InitializeTable(UInt64 polynomial)
        {
            if (polynomial == Crc64Iso.Iso3309Polynomial && Crc64Iso.Table != null)
                return Crc64Iso.Table;

            var createTable = CreateTable(polynomial);

            if (polynomial == Crc64Iso.Iso3309Polynomial)
                Crc64Iso.Table = createTable;

            return createTable;
        }

        protected static ulong[] CreateTable(ulong polynomial)
        {
            var createTable = new UInt64[256];
            for (var i = 0; i < 256; ++i)
            {
                var entry = (UInt64)i;
                for (var j = 0; j < 8; ++j)
                    if ((entry & 1) == 1)
                        entry = (entry >> 1) ^ polynomial;
                    else
                        entry = entry >> 1;
                createTable[i] = entry;
            }
            return createTable;
        }
    }

    public class Crc64Iso : Crc64
    {
        internal static UInt64[] Table;

        public const UInt64 Iso3309Polynomial = 0xD800000000000000;

        public Crc64Iso()
            : base(Iso3309Polynomial)
        {
        }

        public Crc64Iso(UInt64 seed)
            : base(Iso3309Polynomial, seed)
        {
        }

        public static UInt64 Compute(byte[] buffer)
        {
            return Compute(DefaultSeed, buffer);
        }

        public static UInt64 Compute(UInt64 seed, byte[] buffer)
        {
            if (Table == null)
                Table = CreateTable(Iso3309Polynomial);

            return CalculateHash(seed, Table, buffer, 0, buffer.Length);
        }
    }
}


namespace FindSimilar
{
    class CSV_Reader
    {
        private string m_path;
        public CSV_Reader(string path)
        {
            m_path = path;
        }

        public IEnumerable<string[]> Next()
        {
            using (var rdr = new System.IO.StreamReader(m_path))
            {
                string header = rdr.ReadLine();

                while (!rdr.EndOfStream)
                {
                    string line = rdr.ReadLine();
                    if (line == null) break;

                    string[] cols = line.Split(',');
                    yield return cols;
                }
            }
        }
    }


    class Program
    {
        private static string Bytes2Hex(byte[] hashBytes)
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < hashBytes.Length; i++)
            {
                sb.Append(hashBytes[i].ToString("X2"));
            }
            return sb.ToString();
        }

        //static DamienG.Security.Cryptography.Crc64Iso crc64 = new DamienG.Security.Cryptography.Crc64Iso();

        static UInt64 CalcHash(string input)
        {
            byte[] bx = System.Text.Encoding.ASCII.GetBytes(input);
            UInt64 h = DamienG.Security.Cryptography.Crc64Iso.Compute(bx);
            return h;
        }


        static void Main(string[] args)
        {
            //string train_path = "e:/HackerEarth/3/input/train.csv";
            string train_path = "/home/eek/polygon/HackerEarch/3/input/train.csv";
            // ID,datetime,siteid,offerid,category,merchant,countrycode,browserid,devid,click

            //string test_path = "e:/HackerEarth/3/input/test.csv";
            string test_path = "/home/eek/polygon/HackerEarch/3/input/test.csv";
            // ID,datetime,siteid,offerid,category,merchant,countrycode,browserid,devid


            int train_samples = 0;
            HashSet<UInt64> train_hashes = new HashSet<UInt64>();

            Console.WriteLine("Loading data from {0}...", train_path);
            CSV_Reader rdr_train = new CSV_Reader(train_path);
            foreach (string[] cols in rdr_train.Next())
            {
                string train_cols = String.Join("|", Enumerable.Range(1, 8).Select(z => cols[z]));
                UInt64 hash = CalcHash(train_cols);
                train_hashes.Add(hash);
                train_samples++;
            }

            Console.WriteLine("Done, {0} train samples, {1} unique crc64 hashes", train_samples, train_hashes.Count);


            Console.WriteLine("Reading test samples from {0}", test_path);
            int test_samples = 0;
            int hits = 0;
            CSV_Reader rdr_test = new CSV_Reader(test_path);
            foreach (string[] cols in rdr_test.Next())
            {
                string test_cols = String.Join("|", Enumerable.Range(1, 8).Select(z => cols[z]));
                UInt64 hash = CalcHash(test_cols);
                if (train_hashes.Contains(hash))
                {
                    hits++;
                }
                test_samples++;
            }

            Console.WriteLine( "{0} test samples, {1} has duplicates in train dataset", test_samples, hits );

            return;
        }
    }
}
