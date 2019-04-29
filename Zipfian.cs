using java.lang;

namespace Distributions
{
    class Zipfian
    {
        private float skew = 1; // Default value is 1
        private long N; // to be initialized by the constructor
        private double den; // to be calculated during the object instantiation
        private double D; // constant value for the Newton iteration process

        public Zipfian(float skew, long N)
        {
            this.skew = skew;
            this.N = N;
            if (this.skew != 1)
            {
                this.den = (Math.pow(N, 1 - skew) - 1) / (1 - skew) + 0.5 +
                    Math.pow(N, -skew) / 2 + skew / 12 -
                    Math.pow(N, -1 - skew) * skew / 12;
                this.D = 12 * (Math.pow(N, -skew + 1) - 1) / (1 - skew) + 6 +
                    6 * Math.pow(N, -skew) + skew - skew * Math.pow(N, -skew - 1);
            }
            else
            {
                this.den = Math.log(N) + 0.5 + (0.5 / N) + (1.0 / 12.0) - (Math.pow(N, -2) / 12.0);
                this.D = 12 * Math.log(N) + 6 + (6.0 / N) + 1 - Math.pow(N, -2);
            }
        }

        public double zipfCDF(double k)
        {
            if (k > this.N || k < 1)
                throw new IllegalArgumentException("K must be between 1 and N");

            double num = 0;
            if (this.skew != 1)
            {
                num = (Math.pow(k, 1 - this.skew) - 1) / (1 - this.skew) + 0.5 + 
                    Math.pow(k, -this.skew) * 0.5 + this.skew / 12 - 
                    Math.pow(k, -1 - this.skew) * this.skew / 12;
            }
            else
            {
                num = Math.log(k) + 0.5 + (0.5 / k) + (1.0 / 12.0) - Math.pow(k, -2) / 12;
            }

            return num / this.den;
        }

        public long zipfInvCDF(double p)
        {
            if (p > 1 || p < 0)
                throw new IllegalArgumentException("probability p must be between 0 and 1");

            double tol = 0.01; // if rate of change is below tolerance then stop
            double x = this.N / 2.0; // starting value of x (x0)
            double pD = p * this.D;
            
            while (true)
            {
                double m = Math.pow(x, -this.skew - 2);   // x ^ ( -s - 2) for all values of s
                double mx = m * x; // x ^ ( -s - 1) for all values of s
                double mxx = mx * x; // x ^ ( -s) for all values of s 
                double mxxx = mxx * x; // x ^ ( -s + 1), will not be used when s = 1
                double num, den;
                if (this.skew != 1)
                {
                    num = 12 * (mxxx - 1) / (1 - this.skew) + 6 + 6 * mxx + this.skew -
                        (this.skew * mx) - pD;
                    den = 12 * mxx - (6 * this.skew * mx) + (m * this.skew * (this.skew + 1));
                }
                else
                {
                    num = 12 * Math.log(x) + 6 + 6.0 / x + 1 - Math.pow(x, -2) - pD;
                    den = 12 / x - 6 * Math.pow(x, -2) + 2 * Math.pow(x, -3);
                }
                double nextX = Math.max(1, x - num / den);
                if (Math.abs(nextX - x) <= tol) return Math.round(nextX);
                x = nextX;
            }
        }
    }
}
