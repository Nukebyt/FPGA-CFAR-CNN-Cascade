// log2_lzc.v
// log2 of a non-negative fixed-point CODE, by leading-zero-count + mantissa
// ROM -- the exact construction _common/log2_fixed.m models:
//     e = position of the MSB                    (a priority encoder)
//     m = code / 2^e, in [1,2)                    (a barrel shift)
//     log2(code) = e + mant_lut[round((m-1)*n)]
//     log2(value) = log2(code) - fracBits
//
// code==0 clears `valid` (log2 undefined) instead of returning a silently
// wrong finite value -- mirrors log2_fixed.m's -Inf convention, which every
// detector's validity test relies on.
//
// 2-cycle latency: combinational LZC+normalise -> register -> ROM read +
// combine -> register. Output format is fixed at OUT_FRAC=16 (matching the
// mantissa ROM's own Q0.16 precision exactly, so no further rescale multiply
// is needed); OUT_INT is a parameter since callers differ in how large
// log2(value) can get (GenGamma/Burr's log2|s| spans about -12..+1.4, G0's
// log2(c2) about -12..+1.3 -- 5 integer bits covers every one with margin).
module log2_lzc #(
    parameter integer WIDTH     = 26,  // code width (magnitude, unsigned)
    parameter integer FRACBITS  = 24,  // code's own fractional bit count
    parameter integer MANT_BITS = 10,  // mantissa ROM address bits (1024 entries)
    parameter integer OUT_INT   = 5,
    parameter MANT_HEX = "lut/shared/log2_mant_lut.hex"
) (
    input  wire                              clk,
    input  wire [WIDTH-1:0]                  code,
    output reg                               valid,
    output reg  signed [OUT_INT+16:0]        lg     // log2(value), Q(OUT_INT).16 signed
);
    localparam integer MANT_FRAC = 16; // log2_mant_lut.mat format
    localparam integer EW = $clog2(WIDTH);
    localparam integer OUTW = OUT_INT + 17;

    reg [15:0] mant_rom [0:(1<<MANT_BITS)-1];
    initial $readmemh(MANT_HEX, mant_rom);

    // ---- combinational: priority encoder + normalise -----------------------
    integer k;
    reg                      found_c;
    reg [EW-1:0]             e_c;
    reg [WIDTH-1:0]          shifted_c;
    reg [MANT_BITS-1:0]      mant_addr_c;
    reg [WIDTH+MANT_BITS-1:0] mant_scaled_c;

    always @(*) begin
        found_c = 1'b0;
        e_c = {EW{1'b0}};
        for (k = WIDTH-1; k >= 0; k = k-1) begin
            if (!found_c && code[k]) begin
                e_c = k[EW-1:0];
                found_c = 1'b1;
            end
        end
        shifted_c = found_c ? (code << (WIDTH-1-e_c)) : {WIDTH{1'b0}};
        // fractional part of the mantissa (m-1) sits in shifted_c[WIDTH-2:0],
        // at scale 2^(WIDTH-1); round to a MANT_BITS address.
        mant_scaled_c = ({{MANT_BITS{1'b0}}, shifted_c[WIDTH-2:0]} << MANT_BITS)
                        + (1 <<< (WIDTH-2));
        mant_addr_c = mant_scaled_c >> (WIDTH-1);
    end

    // ---- stage 1 register ---------------------------------------------------
    reg                 found_1;
    reg [EW-1:0]        e_1;
    reg [MANT_BITS-1:0] mant_addr_1;
    always @(posedge clk) begin
        found_1     <= found_c;
        e_1         <= e_c;
        mant_addr_1 <= mant_addr_c;
    end

    // ---- stage 2: ROM read + combine ----------------------------------------
    wire [15:0] mant_val = mant_rom[mant_addr_1];              // Q0.16 unsigned
    wire signed [OUTW-1:0] e_scaled    = $signed({{(OUTW-EW){1'b0}}, e_1}) <<< 16;
    wire signed [OUTW-1:0] mant_scaled = $signed({1'b0, mant_val});
    wire signed [OUTW-1:0] frac_const  = FRACBITS <<< 16;

    always @(posedge clk) begin
        valid <= found_1;
        lg    <= e_scaled + mant_scaled - frac_const;
    end
endmodule
