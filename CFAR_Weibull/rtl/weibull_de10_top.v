// weibull_de10_top.v -- standalone DE10-Standard board bring-up wrapper for
// weibull_top_new.v. Direct copy of gengamma_de10_top.v's proven template
// (see that file's header for the full D20 gap-free/frame-reset rationale)
// with the core instantiation swapped -- part of "extending to the other 4
// detectors" (DE10_BRINGUP.md's own open item), for on-board "5 model
// switching" (reprogram via quartus_pgm to switch which detector's bitstream
// is loaded; see DE10_BRINGUP.md).
//
// HONEST NOTE ON THIS DEMO IMAGE'S BEHAVIOR FOR WEIBULL SPECIFICALLY: the
// shared 40x48 demo image (gengamma_demo_image.hex, reused verbatim -- the
// pixel data doesn't depend on which detector processes it) produces GenGamma
// 7 detections at pfa_sel=1, but Weibull's OWN golden vectors
// (CFAR_Weibull/rtl/tb/top_expected.txt) show 0 at that same pfa_sel -- a
// real, verified result (different detectors' thresholds genuinely differ
// on the same data), not a demo bug. LEDR[0] should be expected OFF at
// SW[1:0]=01; it is not a sign of a broken bring-up. weibull_de10_top_tb.v
// checks against this actual (0-detection) golden behavior, not an invented
// "should detect" expectation.
module weibull_de10_top #(
    parameter integer DISPLAY_CYCLES = 50_000_000, // ~1s pause between frames at 50 MHz
    parameter integer RESET_HOLD     = 8            // cycles to hold core_rstn low between frames
) (
    input  wire        CLOCK_50,
    input  wire [1:0]  KEY,   // KEY[0]: board reset (active low, idle=1)
    input  wire [1:0]  SW,    // pfa_sel

    output wire [3:0]  LEDR   // [0]=detect this frame (latched, shown during ST_DISPLAY)
                               // [1]=frame-done blink (one pulse per loop)
                               // [2]=streaming (high while feeding pixels, full speed)
                               // [3]=heartbeat (design alive / clocking)
);
    localparam integer IMG_W = 48, IMG_H = 40;
    localparam integer N_PIX = IMG_W*IMG_H;
    localparam integer ADDR_W = 11; // ceil(log2(1920)) = 11
    localparam integer DRAIN_CYCLES = 32;

    wire board_rstn = KEY[0];

    reg [1:0] pfa_sel_r;
    always @(posedge CLOCK_50 or negedge board_rstn)
        if (!board_rstn) pfa_sel_r <= 2'b00;
        else             pfa_sel_r <= SW;

    reg [7:0] img_rom [0:N_PIX-1];
    initial $readmemh("gengamma_demo_image.hex", img_rom);

    localparam ST_RESET = 3'd0, ST_STREAM = 3'd1, ST_DRAIN = 3'd2, ST_DISPLAY = 3'd3;
    reg [2:0]  state;
    reg [ADDR_W-1:0] pix_addr;
    reg [3:0]  reset_cnt;
    reg [5:0]  drain_cnt;
    reg [31:0] display_cnt;
    reg        core_rstn;
    reg        pixel_in_valid;
    reg [7:0]  pixel_in;

    wire detect_valid, detect;
    wire signed [16:0] T_log_code;

    weibull_top_new #(
        .SLI(17), .GUARD(13), .IMG_WIDTH(IMG_W), .IMG_HEIGHT(IMG_H),
        .LUT_ROOT("F:/Projects/CFAR/lut")
    ) core (
        .clk(CLOCK_50), .rstn(core_rstn),
        .pixel_in_valid(pixel_in_valid), .pixel_in(pixel_in),
        .pfa_sel(pfa_sel_r),
        .detect_valid(detect_valid), .detect(detect), .T_log_code(T_log_code)
    );

    reg frame_done_pulse;
    reg detect_latch, detect_seen_this_frame;

    always @(posedge CLOCK_50 or negedge board_rstn) begin
        if (!board_rstn) begin
            state <= ST_RESET; pix_addr <= 0; reset_cnt <= 0; drain_cnt <= 0; display_cnt <= 0;
            core_rstn <= 1'b0; pixel_in_valid <= 1'b0; pixel_in <= 8'd0;
            frame_done_pulse <= 1'b0; detect_latch <= 1'b0; detect_seen_this_frame <= 1'b0;
        end else begin
            frame_done_pulse <= 1'b0;
            if (detect_valid && detect) detect_seen_this_frame <= 1'b1;

            case (state)
                ST_RESET: begin
                    pixel_in_valid <= 1'b0;
                    if (reset_cnt < RESET_HOLD) begin
                        core_rstn <= 1'b0;
                        reset_cnt <= reset_cnt + 4'd1;
                    end else begin
                        core_rstn <= 1'b1;
                        pix_addr  <= 0;
                        detect_seen_this_frame <= 1'b0;
                        state     <= ST_STREAM;
                    end
                end
                ST_STREAM: begin
                    pixel_in_valid <= 1'b1;
                    pixel_in       <= img_rom[pix_addr];
                    if (pix_addr == N_PIX-1) begin
                        pixel_in_valid <= 1'b0;
                        drain_cnt      <= 0;
                        state          <= ST_DRAIN;
                    end else begin
                        pix_addr <= pix_addr + 1'b1;
                    end
                end
                ST_DRAIN: begin
                    pixel_in_valid <= 1'b0;
                    if (drain_cnt < DRAIN_CYCLES) begin
                        drain_cnt <= drain_cnt + 6'd1;
                    end else begin
                        detect_latch     <= detect_seen_this_frame;
                        frame_done_pulse <= 1'b1;
                        display_cnt      <= 0;
                        state            <= ST_DISPLAY;
                    end
                end
                ST_DISPLAY: begin
                    if (display_cnt < DISPLAY_CYCLES-1) begin
                        display_cnt <= display_cnt + 32'd1;
                    end else begin
                        reset_cnt <= 0;
                        state     <= ST_RESET;
                    end
                end
                default: state <= ST_RESET;
            endcase
        end
    end

    reg [25:0] heartbeat;
    always @(posedge CLOCK_50 or negedge board_rstn)
        if (!board_rstn) heartbeat <= 0;
        else             heartbeat <= heartbeat + 1'b1;

    assign LEDR[0] = detect_latch;
    assign LEDR[1] = frame_done_pulse;
    assign LEDR[2] = (state == ST_STREAM);
    assign LEDR[3] = heartbeat[25];
endmodule
