// event_fifo.v -- small synchronous FIFO for candidate events (RAM-based, registered read).
// Read protocol: assert rd_en for one cycle when !empty; dout is valid on the NEXT cycle (dout_valid).
`timescale 1ns/1ps
module event_fifo #(
    parameter integer WIDTH = 20,
    parameter integer AW    = 10            // depth = 2^AW
) (
    input  wire             clk,
    input  wire             rstn,
    input  wire             wr_en,
    input  wire [WIDTH-1:0] din,
    input  wire             rd_en,
    output reg  [WIDTH-1:0] dout,
    output reg              dout_valid,
    output wire             empty,
    output reg  [AW:0]      count,
    output reg              overflow        // sticky: an event was dropped because the FIFO was full
);
    reg [WIDTH-1:0] mem [0:(1<<AW)-1];
    reg [AW-1:0] wp, rp;
    wire full = (count == (1 << AW));
    assign empty = (count == 0);
    wire do_wr = wr_en & ~full;
    wire do_rd = rd_en & ~empty;

    always @(posedge clk) begin
        if (do_wr) mem[wp] <= din;
        dout <= mem[rp];
    end
    always @(posedge clk) begin
        if (!rstn) begin
            wp <= 0; rp <= 0; count <= 0; overflow <= 1'b0; dout_valid <= 1'b0;
        end else begin
            dout_valid <= do_rd;
            if (do_wr) wp <= wp + 1'b1;
            if (do_rd) rp <= rp + 1'b1;
            if (wr_en & full) overflow <= 1'b1;
            case ({do_wr, do_rd})
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
                default: ;
            endcase
        end
    end
endmodule
