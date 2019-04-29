using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TFFiles
{
    class TFFilesProcessor
    {
        private string tffilepath;
        private FileInfo[] tffilesinfo;
        private BinaryReader[] tffilesreader;
        private int noofdocs = 0;

        // Initialize the instance and read all the *.tf file names into a list of strings
        public TFFilesProcessor(string tffilepath)
        {
            this.tffilepath = tffilepath;
            DirectoryInfo d = new DirectoryInfo(this.tffilepath);
            this.tffilesinfo = d.GetFiles("*.tf");
            this.tffilesreader = new BinaryReader[this.tffilesinfo.Length];
            for (int i = 0; i < this.tffilesreader.Length; i++)
                this.tffilesreader[i] = new BinaryReader(this.tffilesinfo[i].Open(FileMode.Open, FileAccess.Read, FileShare.Read));
            this.getfilecount();
        }


        private void getfilecount()
        {
            this.noofdocs = 0;
            for (int i = 0; i < this.tffilesreader.Length; i++)
            {
                this.tffilesreader[i].BaseStream.Seek(0L, SeekOrigin.Begin);
                this.noofdocs += this.tffilesreader[i].ReadInt32();
            }
        }
    }
}
